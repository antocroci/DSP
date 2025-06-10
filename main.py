#!/usr/bin/env python3
"""
Análisis DOA/TDOA

"""

import argparse
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import Dict, List, Optional

# Importar módulos propios

from simulacion import SimuladorDOA, crear_senal_prueba
from tdoa import EstimadorTDOA
from doa import EstimadorDOA
from evaluacion import EvaluadorDOA


print("=== SIMULADOR DOA CON RUIDO AMBIENTE - PRUEBA ===")

print("\n2. PARÁMETROS ACÚSTICOS:")
ambiente_tipo = input("   Tipo de ambiente (1=Anecoico, 2=Reverberante) [default: 2]: ") or "2"

# Crear simulador
sim = SimuladorDOA(fs=16000)

if ambiente_tipo == "1":
    # Ambiente anecoico CON ruido ambiente
    sim.simular_ambiente_anecoico(
        room_size=[float(input("   Ancho del recinto (X) en metros [default: 6.0]: ") or 6.0), 
                    float(input("   Largo del recinto (Y) en metros [default: 4.0]: ") or 4.0),
                    float(input("   Altura del recinto (Z) en metros [default: 3.0]: ") or 3.0)]
    )
else:
    # Ambiente reverberante CON ruido ambiente
    sim.simular_ambiente_reverberante(
        room_size=[float(input("   Ancho del recinto (X) en metros [default: 6.0]: ") or 6.0), 
                    float(input("   Largo del recinto (Y) en metros [default: 4.0]: ") or 4.0),
                    float(input("   Altura del recinto (Z) en metros [default: 3.0]: ") or 3.0)], 
        rt60= float(input("   RT60 en segundos [default: 0.3]: ") or 0.3),
        ruido_ambiente=True,
        nivel_ruido_db=-40,  # Ruido ambiente moderado
        tipo_ruido='rosa'      # Ruido rosa (más realista)
    )
    
# Crear señal de prueba
signal = crear_senal_prueba("tono", duracion=1.0)

# Configuración de fuentes
print("\n5. FUENTES SONORAS:")
azimuth_real = float(input(f"     Azimuth en grados [default: 60]: ") or 60.0)
distancia_real = float(input("     Distancia en metros [default: 2.0]: ") or 2.0)
elevacion_real = float(input("     Elevación en grados [default: 0.0]: ") or 0.0)

sim.agregar_fuente(signal, azimuth=azimuth_real, distance=distancia_real, elevation=elevacion_real)

# Simular propagación
sim.simular_propagacion(agregar_ruido=True, snr_db=20)
    
# Guardar y visualizar
### sim.guardar_senales(nombre_experimento="test_30deg_ruido_rosa")
sim.visualizar_setup()
    
# Visualizar geometría detallada
sim.visualizar_geometria_detallada()
    
print("\n¡Simulación completada exitosamente!")

# TDOA
print("\n=== TESTING MÓDULO TDOA ===")

# Crear instancia del estimador TDOA
estimador_tdoa = EstimadorTDOA(fs=sim.fs)

# Obtener señales de los micrófonos simulados
mic_signals = sim.signals['mic_signals']
num_mics = mic_signals.shape[0]

print(f"Analizando señales de {num_mics} micrófonos...")

# Preguntar al usuario qué método TDOA usar
print("\nMétodos disponibles:")
print("1. Correlación cruzada clásica")
print("2. GCC básico")
print("3. GCC-PHAT (recomendado)")
metodo_seleccionado = input("Seleccione método [default: 3]: ") or "3"

if metodo_seleccionado == "1":
    metodo = "correlacion"
elif metodo_seleccionado == "2":
    metodo = "gcc"
else:
    metodo = "gcc_phat"

print(f"\nUsando método: {metodo}")

# Estimar TDOA para todos los pares de micrófonos con el micrófono 1 como referencia
resultados_tdoa = estimador_tdoa.estimar_tdoa_array(mic_signals, referencia=0, metodo=metodo)

print("\nResultados TDOA:")
for par, resultado in resultados_tdoa.items():
    tdoa_ms = resultado['tdoa_seconds'] * 1000  # Convertir a milisegundos
    print(f"  {par}: {tdoa_ms:.3f} ms (confianza: {resultado['confidence']:.3f})")

# Calcular DOA usando los TDOAs estimados
spacing = sim.array_geometry['spacing']  # Obtener espaciado del array
print(f"\nCalculando DOA con espaciado de micrófonos: {spacing*100:.1f} cm")

resultados_doa = estimador_tdoa.calcular_doa_lineal(resultados_tdoa, spacing)

print("\nResultados DOA:")
angulos_estimados = []
for par, resultado in resultados_doa.items():
    if resultado['valido']:
        angulo = resultado['angulo_deg']
        angulos_estimados.append(angulo)
        print(f"  {par}: {angulo:.2f}° (válido: {resultado['valido']})")
    else:
        print(f"  {par}: Estimación no válida - {resultado.get('error', 'Error desconocido')}")

# Calcular ángulo promedio si hay estimaciones válidas
if angulos_estimados:
    angulo_promedio = np.mean(angulos_estimados)
    print(f"\nÁngulo DOA promedio estimado: {angulo_promedio:.2f}°")
    print(f"Ángulo real de la fuente: {azimuth_real:.2f}°")
    print(f"Error absoluto: {abs(angulo_promedio - azimuth_real):.2f}°")
else:
    print("\nNo se pudieron obtener estimaciones válidas de DOA")

# Visualizar correlación para un par de micrófonos
print("\nVisualizando correlación entre micrófonos 1 y 2...")
par_visualizar = list(resultados_tdoa.keys())[0]  # Primer par
estimador_tdoa.visualizar_correlacion(
    resultados_tdoa[par_visualizar], 
    titulo=f"Correlación usando {metodo}"
)

# Comparar diferentes métodos para un par de micrófonos
print("\n¿Desea comparar diferentes métodos TDOA? (s/n) [default: n]: ")
comparar = input() or "n"

if comparar.lower() == "s":
    print("Comparando métodos TDOA para micrófonos 1 y 2...")
    metodos_comparar = ['correlacion', 'gcc', 'gcc_phat']
    resultados_comparacion = estimador_tdoa.comparar_metodos(
        mic_signals[0], 
        mic_signals[1], 
        metodos=metodos_comparar
    )
    
    # Mostrar resultados de la comparación
    print("\nResultados de la comparación:")
    for metodo, resultado in resultados_comparacion.items():
        if resultado:
            tdoa_ms = resultado['tdoa_seconds'] * 1000
            print(f"  {metodo}: {tdoa_ms:.3f} ms (confianza: {resultado['confidence']:.3f})")
    
    # Visualizar comparación
    plt.figure(figsize=(12, 8))
    for i, (metodo, resultado) in enumerate(resultados_comparacion.items()):
        if resultado:
            plt.subplot(len(resultados_comparacion), 1, i+1)
            lags_ms = resultado['lags'] * 1000 / sim.fs  # Convertir a ms
            plt.plot(lags_ms, resultado['correlation_normalized'])
            plt.axvline(x=resultado['tdoa_seconds']*1000, color='red', linestyle='--',
                       label=f'TDOA = {resultado["tdoa_seconds"]*1000:.2f} ms')
            plt.title(f'Método: {metodo}')
            plt.grid(True, alpha=0.3)
            plt.legend()
    
    plt.tight_layout()
    plt.show()

print("\n=== Análisis TDOA completado ===")
