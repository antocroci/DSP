"""

#!/usr/bin/env python3
"""
Main Script para Análisis DOA/TDOA
Integra todos los módulos para realizar análisis completos de estimación de dirección de arribo
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
try:
    from simulacion import SimuladorDOA, crear_senal_prueba
    from tdoa import EstimadorTDOA
    from doa import EstimadorDOA
    from evaluacion import EvaluadorDOA
except ImportError as e:
    print(f"Error importando módulos: {e}")
    print("Asegúrate de que todos los archivos .py estén en el mismo directorio")
    sys.exit(1)

class MainDOA:
    """
    Clase principal para ejecutar análisis DOA/TDOA
    """
    
    def __init__(self):
        """
        Inicializa el sistema principal
        """
        self.simulador = None
        self.estimador_tdoa = None
        self.estimador_doa = None
        self.evaluador = None
        self.resultados = {}
        
    def configurar_sistema(self, fs: int = 16000, c: float = 343.0):
        """
        Configura todos los módulos del sistema
        """
        print("Configurando sistema DOA/TDOA...")
        self.simulador = SimuladorDOA(fs=fs)
        self.estimador_tdoa = EstimadorTDOA(fs=fs, c=c)
        self.estimador_doa = EstimadorDOA(c=c)
        self.evaluador = EvaluadorDOA()
        print("✓ Sistema configurado correctamente")
    
    def ejecutar_simulacion_simple(self, 
                                  angulo: float = 30.0,
                                  distancia: float = 2.0,
                                  snr_db: float = 20.0,
                                  tipo_ambiente: str = 'anecoico',
                                  tipo_senal: str = 'tono',
                                  duracion: float = 1.0) -> Dict:
        """
        Ejecuta una simulación simple con parámetros específicos
        """
        print(f"\n=== SIMULACIÓN SIMPLE ===")
        print(f"Ángulo: {angulo}°, Distancia: {distancia}m, SNR: {snr_db}dB")
        print(f"Ambiente: {tipo_ambiente}, Señal: {tipo_senal}")
        
        # Crear señal de prueba
        signal = crear_senal_prueba(tipo_senal, duracion, self.simulador.fs)
        
        # Configurar ambiente
        if tipo_ambiente == 'anecoico':
            self.simulador.simular_ambiente_anecoico()
        elif tipo_ambiente == 'reverberante':
            self.simulador.simular_ambiente_reverberante(rt60=0.3)
        else:
            raise ValueError(f"Tipo de ambiente desconocido: {tipo_ambiente}")
        
        # Agregar fuente y simular
        self.simulador.agregar_fuente(signal, azimuth=angulo, distance=distancia)
        self.simulador.simular_propagacion(agregar_ruido=True, snr_db=snr_db)
        
        # Calcular TDOAs reales (teóricos)
        spacing = self.simulador.array_geometry['spacing']
        tdoa_real = spacing * np.sin(np.deg2rad(angulo)) / self.estimador_tdoa.c
        
        tdoas_reales = {
            'mic_1_mic_2': tdoa_real,
            'mic_1_mic_3': 2 * tdoa_real,
            'mic_1_mic_4': 3 * tdoa_real
        }
        
        # Estimar TDOAs con diferentes métodos
        metodos_tdoa = ['correlacion', 'gcc_phat', 'gcc_scot']
        resultados_tdoa = {}
        
        for metodo in metodos_tdoa:
            try:
                tdoas_estimados = self.estimador_tdoa.estimar_tdoa_array(
                    self.simulador.signals['mic_signals'], metodo=metodo
                )
                resultados_tdoa[metodo] = tdoas_estimados
            except Exception as e:
                print(f"Error con método {metodo}: {e}")
                resultados_tdoa[metodo] = None
        
        # Evaluar métodos TDOA
        evaluacion_tdoa = self.evaluador.evaluar_metodo_tdoa(
            self.estimador_tdoa,
            self.simulador.signals['mic_signals'],
            tdoas_reales,
            metodos_tdoa
        )
        
        # Seleccionar mejor método TDOA
        mejor_metodo_tdoa = min(evaluacion_tdoa.keys(),
                               key=lambda k: evaluacion_tdoa[k].get('error_medio_ms', float('inf')))
        
        print(f"Mejor método TDOA: {mejor_metodo_tdoa}")
        
        # Estimar DOA usando mejor método TDOA
        tdoas_mejor = resultados_tdoa[mejor_metodo_tdoa]
        if tdoas_mejor:
            angulos_individuales = self.estimador_doa.calcular_angulo_arribo(
                tdoas_mejor, spacing
            )
            
            # Promediar ángulos con diferentes métodos
            metodos_doa = ['circular', 'ponderado']
            resultados_doa = {}
            
            for metodo in metodos_doa:
                try:
                    angulo_promedio = self.estimador_doa.promediar_angulos(
                        angulos_individuales, metodo=metodo
                    )
                    resultados_doa[metodo] = angulo_promedio
                except Exception as e:
                    print(f"Error con método DOA {metodo}: {e}")
                    resultados_doa[metodo] = None
            
            # Evaluar métodos DOA
            evaluacion_doa = self.evaluador.evaluar_metodo_doa(
                self.estimador_doa, tdoas_mejor, angulo, spacing, metodos_doa
            )
        else:
            angulos_individuales = {}
            resultados_doa = {}
            evaluacion_doa = {}
        
        # Compilar resultados
        resultado_completo = {
            'parametros': {
                'angulo_real': angulo,
                'distancia': distancia,
                'snr_db': snr_db,
                'tipo_ambiente': tipo_ambiente,
                'tipo_senal': tipo_senal,
                'spacing': spacing
            },
            'tdoas_reales': tdoas_reales,
            'resultados_tdoa': resultados_tdoa,
            'evaluacion_tdoa': evaluacion_tdoa,
            'angulos_individuales': angulos_individuales,
            'resultados_doa': resultados_doa,
            'evaluacion_doa': evaluacion_doa,
            'mejor_metodo_tdoa': mejor_metodo_tdoa
        }
        
        # Mostrar resultados
        self._mostrar_resultados_simple(resultado_completo)
        
        return resultado_completo
    
    def ejecutar_analisis_parametrico(self, 
                                     parametros: Dict,
                                     guardar_resultados: bool = True) -> Dict:
        """
        Ejecuta análisis paramétrico sistemático
        """
        print(f"\n=== ANÁLISIS PARAMÉTRICO ===")
        print(f"Parámetros a evaluar: {list(parametros.keys())}")
        
        # Ejecutar análisis
        resultados_parametricos = self.evaluador.analisis_parametrico(
            self.simulador, self.estimador_tdoa, self.estimador_doa, parametros
        )
        
        # Generar gráficos para cada parámetro
        for param in parametros.keys():
            if resultados_parametricos[param]:
                print(f"Generando gráficos para {param}...")
                self.evaluador.graficar_resultados(resultados_parametricos, param)
        
        # Guardar resultados si se solicita
        if guardar_resultados:
            filename = f"analisis_parametrico_{self._timestamp()}.json"
            self.evaluador.exportar_resultados(resultados_parametricos, filename)
        
        return resultados_parametricos
    
    def ejecutar_comparacion_metodos(self, 
                                   condiciones_test: List[Dict],
                                   guardar_resultados: bool = True) -> Dict:
        """
        Ejecuta comparación sistemática de métodos
        """
        print(f"\n=== COMPARACIÓN DE MÉTODOS ===")
        print(f"Evaluando {len(condiciones_test)} condiciones diferentes")
        
        resultados_comparacion = {
            'condiciones': [],
            'resumen_tdoa': {},
            'resumen_doa': {},
            'ranking_global': {}
        }
        
        # Evaluar cada condición
        for i, condicion in enumerate(condiciones_test):
            print(f"\nCondición {i+1}/{len(condiciones_test)}: {condicion}")
            
            try:
                resultado = self.ejecutar_simulacion_simple(**condicion)
                resultados_comparacion['condiciones'].append(resultado)
            except Exception as e:
                print(f"Error en condición {i+1}: {e}")
        
        # Analizar resultados globales
        if resultados_comparacion['condiciones']:
            self._analizar_resultados_globales(resultados_comparacion)
        
        # Guardar resultados
        if guardar_resultados:
            filename = f"comparacion_metodos_{self._timestamp()}.json"
            self.evaluador.exportar_resultados(resultados_comparacion, filename)
        
        return resultados_comparacion
    
    def _mostrar_resultados_simple(self, resultado: Dict):
        """
        Muestra resultados de simulación simple
        """
        print(f"\n--- RESULTADOS ---")
        
        # Parámetros
        params = resultado['parametros']
        print(f"Ángulo real: {params['angulo_real']}°")
        print(f"Distancia: {params['distancia']}m")
        print(f"SNR: {params['snr_db']}dB")
        
        # Mejor TDOA
        mejor_tdoa = resultado['mejor_metodo_tdoa']
        eval_tdoa = resultado['evaluacion_tdoa'].get(mejor_tdoa, {})
        if 'error_medio_ms' in eval_tdoa:
            print(f"\nMejor TDOA ({mejor_tdoa}): {eval_tdoa['error_medio_ms']:.3f} ms error")
        
        # Mejor DOA
        eval_doa = resultado['evaluacion_doa']
        if eval_doa:
            mejor_doa = min(eval_doa.keys(), 
                           key=lambda k: eval_doa[k].get('error_absoluto_deg', float('inf')))
            doa_data = eval_doa[mejor_doa]
            if doa_data.get('valido', False):
                print(f"Mejor DOA ({mejor_doa}): {doa_data['angulo_estimado']:.2f}° "
                      f"(error: {doa_data['error_absoluto_deg']:.2f}°)")
    
    def _analizar_resultados_globales(self, resultados: Dict):
        """
        Analiza resultados globales de múltiples condiciones
        """
        print(f"\n--- ANÁLISIS GLOBAL ---")
        
        # Recopilar errores por método
        errores_tdoa = {}
        errores_doa = {}
        
        for resultado in resultados['condiciones']:
            # Errores TDOA
            for metodo, eval_data in resultado['evaluacion_tdoa'].items():
                if 'error_medio_ms' in eval_data:
                    if metodo not in errores_tdoa:
                        errores_tdoa[metodo] = []
                    errores_tdoa[metodo].append(eval_data['error_medio_ms'])
            
            # Errores DOA
            for metodo, eval_data in resultado['evaluacion_doa'].items():
                if eval_data.get('valido', False):
                    if metodo not in errores_doa:
                        errores_doa[metodo] = []
                    errores_doa[metodo].append(eval_data['error_absoluto_deg'])
        
        # Estadísticas TDOA
        print("\nRESUMEN TDOA:")
        for metodo, errores in errores_tdoa.items():
            if errores:
                print(f"  {metodo}: {np.mean(errores):.3f} ± {np.std(errores):.3f} ms")
        
        # Estadísticas DOA
        print("\nRESUMEN DOA:")
        for metodo, errores in errores_doa.items():
            if errores:
                print(f"  {metodo}: {np.mean(errores):.2f} ± {np.std(errores):.2f}°")
        
        # Guardar en resultados
        resultados['resumen_tdoa'] = {
            metodo: {
                'error_medio': float(np.mean(errores)),
                'error_std': float(np.std(errores)),
                'num_evaluaciones': len(errores)
            } for metodo, errores in errores_tdoa.items() if errores
        }
        
        resultados['resumen_doa'] = {
            metodo: {
                'error_medio': float(np.mean(errores)),
                'error_std': float(np.std(errores)),
                'num_evaluaciones': len(errores)
            } for metodo, errores in errores_doa.items() if errores
        }
    
    def _timestamp(self) -> str:
        """
        Genera timestamp para nombres de archivo
        """
        from datetime import datetime
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def menu_interactivo(self):
        """
        Menú interactivo para seleccionar análisis
        """
        while True:
            print("\n" + "="*50)
            print("SISTEMA DE ANÁLISIS DOA/TDOA")
            print("="*50)
            print("1. Simulación simple")
            print("2. Análisis paramétrico")
            print("3. Comparación de métodos")
            print("4. Ejemplo completo")
            print("5. Salir")
            print("-"*50)
            
            try:
                opcion = input("Selecciona una opción (1-5): ").strip()
                
                if opcion == '1':
                    self._menu_simulacion_simple()
                elif opcion == '2':
                    self._menu_analisis_parametrico()
                elif opcion == '3':
                    self._menu_comparacion_metodos()
                elif opcion == '4':
                    self._ejecutar_ejemplo_completo()
                elif opcion == '5':
                    print("¡Hasta luego!")
                    break
                else:
                    print("Opción no válida. Intenta de nuevo.")
                    
            except KeyboardInterrupt:
                print("\n¡Hasta luego!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def _menu_simulacion_simple(self):
        """
        Menú para simulación simple
        """
        print("\n--- SIMULACIÓN SIMPLE ---")
        try:
            angulo = float(input("Ángulo de la fuente (grados) [30]: ") or "30")
            distancia = float(input("Distancia de la fuente (metros) [2.0]: ") or "2.0")
            snr_db = float(input("SNR (dB) [20]: ") or "20")
            
            print("Tipos de ambiente: anecoico, reverberante")
            ambiente = input("Tipo de ambiente [anecoico]: ") or "anecoico"
            
            print("Tipos de señal: tono, chirp, ruido")
            senal = input("Tipo de señal [tono]: ") or "tono"
            
            resultado = self.ejecutar_simulacion_simple(
                angulo=angulo, distancia=distancia, snr_db=snr_db,
                tipo_ambiente=ambiente, tipo_senal=senal
            )
            
            # Preguntar si visualizar
            if input("\n¿Visualizar setup? (s/n) [s]: ").lower() != 'n':
                self.simulador.visualizar_setup()
                
        except ValueError as e:
            print(f"Error en los valores ingresados: {e}")
    
    def _crear_senal_test(self, duracion=0.5, fs=16000):  # Reducir duración por defecto
        """
        Crea señal de prueba estándar (optimizada)
        """
        t = np.linspace(0, duracion, int(duracion * fs))
        return np.sin(2 * np.pi * 1000 * t)  # Tono de 1kHz

    def _menu_analisis_parametrico(self):
        """
        Menú para análisis paramétrico (optimizado)
        """
        print("\n--- ANÁLISIS PARAMÉTRICO ---")
        print("Selecciona parámetros a evaluar:")
        print("1. Ángulos (rápido)")
        print("2. Distancias (rápido)") 
        print("3. SNR (rápido)")
        print("4. Todos (lento)")
        
        opcion = input("Opción [1]: ") or "1"  # Por defecto solo ángulos
        
        parametros = {}
        
        if opcion in ['1', '4']:
            parametros['angulos'] = np.arange(15, 76, 30)  # Menos puntos: 15°, 45°, 75°
        if opcion in ['2', '4']:
            parametros['distancias'] = np.arange(1.0, 4.1, 1.5)  # Menos puntos: 1m, 2.5m, 4m
        if opcion in ['3', '4']:
            parametros['snr_values'] = np.arange(10, 26, 10)  # Menos puntos: 10dB, 20dB
        
        print(f"Ejecutando análisis con: {list(parametros.keys())}")
        self.ejecutar_analisis_parametrico(parametros)
    
    def _menu_comparacion_metodos(self):
        """
        Menú para comparación de métodos
        """
        print("\n--- COMPARACIÓN DE MÉTODOS ---")
        
        # Condiciones de prueba predefinidas
        condiciones = [
            {'angulo': 15, 'distancia': 2.0, 'snr_db': 20, 'tipo_ambiente': 'anecoico'},
            {'angulo': 30, 'distancia': 2.0, 'snr_db': 20, 'tipo_ambiente': 'anecoico'},
            {'angulo': 45, 'distancia': 2.0, 'snr_db': 20, 'tipo_ambiente': 'anecoico'},
            {'angulo': 60, 'distancia': 2.0, 'snr_db': 20, 'tipo_ambiente': 'anecoico'},
            {'angulo': 30, 'distancia': 2.0, 'snr_db': 10, 'tipo_ambiente': 'anecoico'},
            {'angulo': 30, 'distancia': 2.0, 'snr_db': 20, 'tipo_ambiente': 'reverberante'},
        ]
        
        print(f"Evaluando {len(condiciones)} condiciones predefinidas...")
        self.ejecutar_comparacion_metodos(condiciones)
    
    def _ejecutar_ejemplo_completo(self):
        """
        Ejecuta un ejemplo completo demostrativo
        """
        print("\n--- EJEMPLO COMPLETO ---")
        print("Ejecutando análisis completo con múltiples condiciones...")
        
        # Simulación simple
        print("\n1. Simulación simple...")
        resultado_simple = self.ejecutar_simulacion_simple(
            angulo=30, distancia=2.0, snr_db=20
        )
        
        # Análisis paramétrico limitado
        print("\n2. Análisis paramétrico (ángulos)...")
        parametros = {'angulos': [15, 30, 45, 60]}
        resultado_parametrico = self.ejecutar_analisis_parametrico(parametros)
        
        # Comparación de métodos
        print("\n3. Comparación de métodos...")
        condiciones = [
            {'angulo': 30, 'snr_db': 20},
            {'angulo': 30, 'snr_db': 10},
            {'angulo': 45, 'snr_db': 20}
        ]
        resultado_comparacion = self.ejecutar_comparacion_metodos(condiciones)
        
        print("\n¡Ejemplo completo finalizado!")
        print("Revisa los archivos JSON generados para resultados detallados.")

def main():
    """
    Función principal
    """
    parser = argparse.ArgumentParser(description='Sistema de Análisis DOA/TDOA')
    parser.add_argument('--modo', choices=['simple', 'parametrico', 'comparacion', 'interactivo'],
                       default='interactivo', help='Modo de ejecución')
    parser.add_argument('--angulo', type=float, default=30.0, help='Ángulo de la fuente (grados)')
    parser.add_argument('--distancia', type=float, default=2.0, help='Distancia de la fuente (metros)')
    parser.add_argument('--snr', type=float, default=20.0, help='SNR en dB')
    parser.add_argument('--ambiente', choices=['anecoico', 'reverberante'], 
                       default='anecoico', help='Tipo de ambiente')
    parser.add_argument('--senal', choices=['tono', 'chirp', 'ruido'], 
                       default='tono', help='Tipo de señal')
    parser.add_argument('--fs', type=int, default=16000, help='Frecuencia de muestreo')
    parser.add_argument('--no-graficos', action='store_true', help='No mostrar gráficos')
    
    args = parser.parse_args()
    
    # Configurar matplotlib para no mostrar gráficos si se solicita
    if args.no_graficos:
        import matplotlib
        matplotlib.use('Agg')
    
    # Crear y configurar sistema
    sistema = MainDOA()
    sistema.configurar_sistema(fs=args.fs)
    
    try:
        if args.modo == 'simple':
            resultado = sistema.ejecutar_simulacion_simple(
                angulo=args.angulo,
                distancia=args.distancia,
                snr_db=args.snr,
                tipo_ambiente=args.ambiente,
                tipo_senal=args.senal
            )
            
        elif args.modo == 'parametrico':
            parametros = {
                'angulos': np.arange(0, 91, 15),
                'distancias': np.arange(1.0, 5.1, 1.0),
                'snr_values': np.arange(5, 31, 5)
            }
            resultado = sistema.ejecutar_analisis_parametrico(parametros)
            
        elif args.modo == 'comparacion':
            condiciones = [
                {'angulo': 15, 'snr_db': 20},
                {'angulo': 30, 'snr_db': 20},
                {'angulo': 45, 'snr_db': 20},
                {'angulo': 60, 'snr_db': 20}
            ]
            resultado = sistema.ejecutar_comparacion_metodos(condiciones)
            
        elif args.modo == 'interactivo':
            sistema.menu_interactivo()
            
    except KeyboardInterrupt:
        print("\nEjecución interrumpida por el usuario")
    except Exception as e:
        print(f"Error durante la ejecución: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

"""