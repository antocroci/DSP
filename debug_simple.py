
#!/usr/bin/env python3

"""
"""
Script de debugging simple para identificar problemas específicos
"""

import numpy as np
import matplotlib.pyplot as plt

# Importar módulos
from simulacion import SimuladorDOA, crear_senal_prueba
from tdoa import EstimadorTDOA
from doa import EstimadorDOA

def test_basico():
    """
    Test básico para identificar el problema
    """
    print("=== TEST BÁSICO DE DEBUGGING ===")
    
    # Configurar sistema
    fs = 16000
    simulador = SimuladorDOA(fs=fs)
    estimador_tdoa = EstimadorTDOA(fs=fs)
    estimador_doa = EstimadorDOA()
    
    # Crear señal corta para test rápido
    signal = crear_senal_prueba("tono", duracion=0.2, fs=fs)  # Solo 0.2 segundos
    print(f"Señal creada: {len(signal)} muestras")
    
    # Simulación simple
    simulador.simular_ambiente_anecoico()
    simulador.agregar_fuente(signal, azimuth=30, distance=2.0)
    simulador.simular_propagacion(agregar_ruido=True, snr_db=20)
    
    print(f"Simulación completada: {simulador.signals['mic_signals'].shape}")
    
    # Test TDOA individual
    print("\n--- TEST TDOA ---")
    mic1 = simulador.signals['mic_signals'][0, :]
    mic2 = simulador.signals['mic_signals'][1, :]
    
    print(f"Mic1 shape: {mic1.shape}, dtype: {mic1.dtype}")
    print(f"Mic2 shape: {mic2.shape}, dtype: {mic2.dtype}")
    
    try:
        # Test correlación simple
        resultado_corr = estimador_tdoa.estimar_tdoa_par(mic1, mic2, metodo='correlacion')
        print(f"Correlación exitosa: TDOA = {resultado_corr['tdoa_seconds']:.6f}s")
        
        # Test GCC-PHAT
        resultado_gcc = estimador_tdoa.estimar_tdoa_par(mic1, mic2, metodo='gcc_phat')
        print(f"GCC-PHAT exitoso: TDOA = {resultado_gcc['tdoa_seconds']:.6f}s")
        
    except Exception as e:
        print(f"Error en TDOA: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test DOA
    print("\n--- TEST DOA ---")
    try:
        # Crear TDOAs simulados
        spacing = simulador.array_geometry['spacing']
        tdoa_real = spacing * np.sin(np.deg2rad(30)) / estimador_tdoa.c
        
        tdoas_test = {
            'mic_1_mic_2': {
                'tdoa_seconds': tdoa_real,
                'confidence': 0.9,
                'valido': True
            }
        }
        
        angulos = estimador_doa.calcular_angulo_arribo(tdoas_test, spacing)
        print(f"DOA exitoso: {angulos}")
        
    except Exception as e:
        print(f"Error en DOA: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n✓ Test básico completado exitosamente")
    return True

def test_evaluacion():
    """
    Test específico del módulo de evaluación
    """
    print("\n=== TEST EVALUACIÓN ===")
    
    from evaluacion import EvaluadorDOA
    evaluador = EvaluadorDOA()
    
    # Test con datos simples
    try:
        # Test con escalares
        error1 = evaluador.calcular_error(30.5, 30.0, 'absoluto')
        print(f"Error escalar: {error1['error_medio']}")
        
        # Test con listas
        error2 = evaluador.calcular_error([29.5, 30.5], [30.0, 30.0], 'absoluto')
        print(f"Error lista: {error2['error_medio']}")
        
        print("✓ Evaluación funcionando correctamente")
        return True
        
    except Exception as e:
        print(f"Error en evaluación: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Iniciando debugging...")
    
    # Test básico
    if test_basico():
        print("\n" + "="*50)
        # Test evaluación
        test_evaluacion()
    else:
        print("❌ Test básico falló - revisar configuración")

"""