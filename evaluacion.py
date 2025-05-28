import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import pandas as pd
from scipy import stats
import warnings
from pathlib import Path
import json

class EvaluadorDOA:
    """
    Evaluador completo para sistemas DOA/TDOA
    """
    
    def __init__(self):
        """
        Inicializa el evaluador
        """
        self.resultados = {}
        self.metricas = {}
        
    def calcular_error(self, 
                      estimado: Union[float, List[float]], 
                      real: Union[float, List[float]],
                      tipo_error: str = 'absoluto') -> Dict:
        """
        Calcula diferentes tipos de errores
        
        Args:
            estimado: Valor(es) estimado(s)
            real: Valor(es) real(es)
            tipo_error: 'absoluto', 'relativo', 'cuadratico', 'angular'
            
        Returns:
            Diccionario con métricas de error
        """
        # Convertir a arrays numpy
        est = np.array(estimado) if not isinstance(estimado, np.ndarray) else estimado
        real_val = np.array(real) if not isinstance(real, np.ndarray) else real
        
        # Asegurar misma forma
        if est.shape != real_val.shape:
            raise ValueError("Las dimensiones de estimado y real deben coincidir")
        
        if tipo_error == 'absoluto':
            error = np.abs(est - real_val)
        elif tipo_error == 'relativo':
            error = np.abs(est - real_val) / np.abs(real_val) * 100
            error = np.where(np.abs(real_val) < 1e-10, np.inf, error)  # Evitar división por cero
        elif tipo_error == 'cuadratico':
            error = (est - real_val) ** 2
        elif tipo_error == 'angular':
            # Error angular que maneja periodicidad
            diff = est - real_val
            error = np.abs(((diff + 180) % 360) - 180)
        else:
            raise ValueError(f"Tipo de error desconocido: {tipo_error}")
        
        # Calcular estadísticas
        metricas = {
            'error_medio': float(np.mean(error)),
            'error_std': float(np.std(error)),
            'error_max': float(np.max(error)),
            'error_min': float(np.min(error)),
            'error_mediano': float(np.median(error)),
            'rmse': float(np.sqrt(np.mean(error**2))) if tipo_error != 'cuadratico' else float(np.sqrt(np.mean(error))),
            'mae': float(np.mean(np.abs(error))) if tipo_error != 'absoluto' else float(np.mean(error)),
            'num_muestras': len(error),
            'tipo_error': tipo_error
        }
        
        return metricas
    
    def evaluar_metodo_tdoa(self, 
                           estimador_tdoa,
                           signals: np.ndarray,
                           tdoas_reales: Dict,
                           metodos: List[str] = ['correlacion', 'gcc_phat', 'gcc_scot']) -> Dict:
        """
        Evalúa diferentes métodos TDOA
        
        Args:
            estimador_tdoa: Instancia de EstimadorTDOA
            signals: Señales de micrófonos (num_mics x num_samples)
            tdoas_reales: TDOAs verdaderos
            metodos: Lista de métodos a evaluar
            
        Returns:
            Resultados de evaluación por método
        """
        resultados_metodos = {}
        
        for metodo in metodos:
            print(f"Evaluando método TDOA: {metodo}")
            
            try:
                # Estimar TDOAs con el método actual
                tdoas_estimados = estimador_tdoa.estimar_tdoa_array(
                    signals, metodo=metodo
                )
                
                # Calcular errores para cada par
                errores_pares = {}
                for par_key in tdoas_reales.keys():
                    if par_key in tdoas_estimados:
                        tdoa_est = tdoas_estimados[par_key]['tdoa_seconds']
                        tdoa_real = tdoas_reales[par_key]
                        
                        error_data = self.calcular_error(tdoa_est, tdoa_real, 'absoluto')
                        errores_pares[par_key] = {
                            'tdoa_estimado': tdoa_est,
                            'tdoa_real': tdoa_real,
                            'error_absoluto': error_data['error_medio'],
                            'confidence': tdoas_estimados[par_key].get('confidence', 0.0)
                        }
                
                # Métricas globales del método
                errores_absolutos = [e['error_absoluto'] for e in errores_pares.values()]
                confidencias = [e['confidence'] for e in errores_pares.values()]
                
                resultados_metodos[metodo] = {
                    'errores_pares': errores_pares,
                    'error_medio_ms': np.mean(errores_absolutos) * 1000,
                    'error_std_ms': np.std(errores_absolutos) * 1000,
                    'error_max_ms': np.max(errores_absolutos) * 1000,
                    'confidence_media': np.mean(confidencias),
                    'num_estimaciones_validas': len(errores_pares),
                    'metodo': metodo
                }
                
            except Exception as e:
                print(f"Error evaluando método {metodo}: {e}")
                resultados_metodos[metodo] = {
                    'error': str(e),
                    'metodo': metodo,
                    'valido': False
                }
        
        return resultados_metodos
    
    def evaluar_metodo_doa(self, 
                          estimador_doa,
                          tdoas_estimados: Dict,
                          angulo_real: float,
                          spacing: float,
                          metodos_promedio: List[str] = ['circular', 'ponderado']) -> Dict:
        """
        Evalúa diferentes métodos DOA
        """
        resultados_doa = {}
        
        # Calcular ángulos individuales
        angulos_individuales = estimador_doa.calcular_angulo_arribo(
            tdoas_estimados, spacing
        )
        
        for metodo_promedio in metodos_promedio:
            try:
                # Promediar ángulos
                angulo_promedio = estimador_doa.promediar_angulos(
                    angulos_individuales, metodo=metodo_promedio
                )
                
                if angulo_promedio.get('valido', False):
                    # Calcular error angular
                    error_angular = self.calcular_error(
                        angulo_promedio['angulo_promedio_deg'], 
                        angulo_real, 
                        'angular'
                    )
                    
                    resultados_doa[metodo_promedio] = {
                        'angulo_estimado': angulo_promedio['angulo_promedio_deg'],
                        'angulo_real': angulo_real,
                        'error_absoluto_deg': error_angular['error_medio'],
                        'std_estimacion_deg': angulo_promedio['std_deg'],
                        'num_estimaciones': angulo_promedio['num_estimaciones'],
                        'metodo': metodo_promedio,
                        'valido': True
                    }
                else:
                    resultados_doa[metodo_promedio] = {
                        'error': 'Promediado falló',
                        'metodo': metodo_promedio,
                        'valido': False
                    }
                    
            except Exception as e:
                resultados_doa[metodo_promedio] = {
                    'error': str(e),
                    'metodo': metodo_promedio,
                    'valido': False
                }
        
        return resultados_doa
    
    def comparar_algoritmos(self, 
                           resultados_tdoa: Dict,
                           resultados_doa: Dict) -> Dict:
        """
        Compara diferentes algoritmos y genera ranking
        """
        comparacion = {
            'tdoa': {},
            'doa': {},
            'ranking_tdoa': [],
            'ranking_doa': []
        }
        
        # Comparar métodos TDOA
        metodos_tdoa_validos = []
        for metodo, resultado in resultados_tdoa.items():
            if resultado.get('valido', True) and 'error_medio_ms' in resultado:
                metodos_tdoa_validos.append({
                    'metodo': metodo,
                    'error_medio_ms': resultado['error_medio_ms'],
                    'error_std_ms': resultado['error_std_ms'],
                    'confidence_media': resultado['confidence_media']
                })
        
        # Ranking TDOA (menor error es mejor)
        metodos_tdoa_validos.sort(key=lambda x: x['error_medio_ms'])
        comparacion['ranking_tdoa'] = metodos_tdoa_validos
        
        # Comparar métodos DOA
        metodos_doa_validos = []
        for metodo, resultado in resultados_doa.items():
            if resultado.get('valido', False):
                metodos_doa_validos.append({
                    'metodo': metodo,
                    'error_absoluto_deg': resultado['error_absoluto_deg'],
                    'std_estimacion_deg': resultado.get('std_estimacion_deg', 0.0),
                    'num_estimaciones': resultado.get('num_estimaciones', 1)
                })
        
        # Ranking DOA (menor error es mejor)
        metodos_doa_validos.sort(key=lambda x: x['error_absoluto_deg'])
        comparacion['ranking_doa'] = metodos_doa_validos
        
        # Estadísticas de comparación
        if metodos_tdoa_validos:
            mejor_tdoa = metodos_tdoa_validos[0]
            peor_tdoa = metodos_tdoa_validos[-1]
            comparacion['tdoa'] = {
                'mejor_metodo': mejor_tdoa['metodo'],
                'mejor_error_ms': mejor_tdoa['error_medio_ms'],
                'peor_metodo': peor_tdoa['metodo'],
                'peor_error_ms': peor_tdoa['error_medio_ms'],
                'mejora_relativa': (peor_tdoa['error_medio_ms'] - mejor_tdoa['error_medio_ms']) / peor_tdoa['error_medio_ms'] * 100
            }
        
        if metodos_doa_validos:
            mejor_doa = metodos_doa_validos[0]
            peor_doa = metodos_doa_validos[-1]
            comparacion['doa'] = {
                'mejor_metodo': mejor_doa['metodo'],
                'mejor_error_deg': mejor_doa['error_absoluto_deg'],
                'peor_metodo': peor_doa['metodo'],
                'peor_error_deg': peor_doa['error_absoluto_deg'],
                'mejora_relativa': (peor_doa['error_absoluto_deg'] - mejor_doa['error_absoluto_deg']) / peor_doa['error_absoluto_deg'] * 100
            }
        
        return comparacion
    
    def analisis_parametrico(self, 
                            simulador,
                            estimador_tdoa,
                            estimador_doa,
                            parametros: Dict) -> Dict:
        """
        Análisis paramétrico sistemático
        
        Args:
            simulador: Instancia de SimuladorDOA
            estimador_tdoa: Instancia de EstimadorTDOA
            estimador_doa: Instancia de EstimadorDOA
            parametros: Diccionario con rangos de parámetros a evaluar
        """
        resultados_parametricos = {
            'angulos': [],
            'distancias': [],
            'snr_values': [],
            'num_mics': [],
            'spacing_values': []
        }
        
        # Análisis vs ángulo
        if 'angulos' in parametros:
            print("Analizando vs ángulo...")
            angulos_test = parametros['angulos']
            
            for angulo in angulos_test:
                try:
                    # Crear simulación
                    sim = simulador.__class__(fs=simulador.fs)
                    signal = self._crear_senal_test()
                    
                    sim.simular_ambiente_anecoico()
                    sim.agregar_fuente(signal, azimuth=angulo, distance=2.0)
                    sim.simular_propagacion(agregar_ruido=True, snr_db=20)
                    
                    # Evaluar
                    resultado = self._evaluar_simulacion_completa(
                        sim, estimador_tdoa, estimador_doa, angulo
                    )
                    resultado['parametro'] = angulo
                    resultados_parametricos['angulos'].append(resultado)
                    
                except Exception as e:
                    print(f"Error en ángulo {angulo}: {e}")
        
        # Análisis vs distancia
        if 'distancias' in parametros:
            print("Analizando vs distancia...")
            distancias_test = parametros['distancias']
            
            for distancia in distancias_test:
                try:
                    sim = simulador.__class__(fs=simulador.fs)
                    signal = self._crear_senal_test()
                    
                    sim.simular_ambiente_anecoico()
                    sim.agregar_fuente(signal, azimuth=30.0, distance=distancia)
                    sim.simular_propagacion(agregar_ruido=True, snr_db=20)
                    
                    resultado = self._evaluar_simulacion_completa(
                        sim, estimador_tdoa, estimador_doa, 30.0
                    )
                    resultado['parametro'] = distancia
                    resultados_parametricos['distancias'].append(resultado)
                    
                except Exception as e:
                    print(f"Error en distancia {distancia}: {e}")
        
        # Análisis vs SNR
        if 'snr_values' in parametros:
            print("Analizando vs SNR...")
            snr_values = parametros['snr_values']
            
            for snr in snr_values:
                try:
                    sim = simulador.__class__(fs=simulador.fs)
                    signal = self._crear_senal_test()
                    
                    sim.simular_ambiente_anecoico()
                    sim.agregar_fuente(signal, azimuth=30.0, distance=2.0)
                    sim.simular_propagacion(agregar_ruido=True, snr_db=snr)
                    
                    resultado = self._evaluar_simulacion_completa(
                        sim, estimador_tdoa, estimador_doa, 30.0
                    )
                    resultado['parametro'] = snr
                    resultados_parametricos['snr_values'].append(resultado)
                    
                except Exception as e:
                    print(f"Error en SNR {snr}: {e}")
        
        return resultados_parametricos
    
    def _evaluar_simulacion_completa(self, sim, estimador_tdoa, estimador_doa, angulo_real):
        """
        Evalúa una simulación completa
        """
        # Calcular TDOAs reales
        spacing = sim.array_geometry['spacing']
        tdoa_real = spacing * np.sin(np.deg2rad(angulo_real)) / estimador_tdoa.c
        
        tdoas_reales = {
            'mic_1_mic_2': tdoa_real,
            'mic_1_mic_3': 2 * tdoa_real,
            'mic_1_mic_4': 3 * tdoa_real
        }
        
        # Evaluar TDOA
        resultados_tdoa = self.evaluar_metodo_tdoa(
            estimador_tdoa, sim.signals['mic_signals'], tdoas_reales
        )
        
        # Evaluar DOA (usar mejor método TDOA)
        mejor_metodo_tdoa = min(resultados_tdoa.keys(), 
                               key=lambda k: resultados_tdoa[k].get('error_medio_ms', float('inf')))
        
        tdoas_estimados = estimador_tdoa.estimar_tdoa_array(
            sim.signals['mic_signals'], metodo=mejor_metodo_tdoa
        )
        
        resultados_doa = self.evaluar_metodo_doa(
            estimador_doa, tdoas_estimados, angulo_real, spacing
        )
        
        return {
            'tdoa': resultados_tdoa,
            'doa': resultados_doa,
            'angulo_real': angulo_real,
            'spacing': spacing
        }
    
    def _crear_senal_test(self, duracion=1.0, fs=16000):
        """
        Crea señal de prueba estándar
        """
        t = np.linspace(0, duracion, int(duracion * fs))
        return np.sin(2 * np.pi * 1000 * t)  # Tono de 1kHz
    
    def graficar_resultados(self, 
                           resultados_parametricos: Dict,
                           parametro: str,
                           metrica: str = 'error_doa') -> None:
        """
        Genera gráficos de resultados paramétricos
        
        Args:
            resultados_parametricos: Resultados del análisis paramétrico
            parametro: 'angulos', 'distancias', 'snr_values', etc.
            metrica: 'error_tdoa', 'error_doa', 'confidence'
        """
        if parametro not in resultados_parametricos:
            print(f"Parámetro {parametro} no encontrado en resultados")
            return
        
        datos = resultados_parametricos[parametro]
        if not datos:
            print(f"No hay datos para parámetro {parametro}")
            return
        
        # Configurar estilo
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Análisis de Rendimiento vs {parametro.title()}', fontsize=16)
        
        # Extraer datos
        param_values = [d['parametro'] for d in datos]
        
        # Gráfico 1: Error TDOA por método
        ax1 = axes[0, 0]
        metodos_tdoa = set()
        for d in datos:
            metodos_tdoa.update(d['tdoa'].keys())
        
        for metodo in metodos_tdoa:
            errores = []
            for d in datos:
                if metodo in d['tdoa'] and 'error_medio_ms' in d['tdoa'][metodo]:
                    errores.append(d['tdoa'][metodo]['error_medio_ms'])
                else:
                    errores.append(np.nan)
            
            ax1.plot(param_values, errores, 'o-', label=metodo, linewidth=2, markersize=6)
        
        ax1.set_xlabel(parametro.replace('_', ' ').title())
        ax1.set_ylabel('Error TDOA (ms)')
        ax1.set_title('Error TDOA por Método')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Gráfico 2: Error DOA por método
        ax2 = axes[0, 1]
        metodos_doa = set()
        for d in datos:
            metodos_doa.update(d['doa'].keys())
        
        for metodo in metodos_doa:
            errores = []
            for d in datos:
                if metodo in d['doa'] and d['doa'][metodo].get('valido', False):
                    errores.append(d['doa'][metodo]['error_absoluto_deg'])
                else:
                    errores.append(np.nan)
            
            ax2.plot(param_values, errores, 's-', label=metodo, linewidth=2, markersize=6)
        
        ax2.set_xlabel(parametro.replace('_', ' ').title())
        ax2.set_ylabel('Error DOA (°)')
        ax2.set_title('Error DOA por Método')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Gráfico 3: Comparación de mejor método
        ax3 = axes[1, 0]
        mejor_tdoa = []
        mejor_doa = []
        
        for d in datos:
            # Mejor TDOA
            min_error_tdoa = float('inf')
            for metodo, resultado in d['tdoa'].items():
                if 'error_medio_ms' in resultado:
                    if resultado['error_medio_ms'] < min_error_tdoa:
                        min_error_tdoa = resultado['error_medio_ms']
            mejor_tdoa.append(min_error_tdoa if min_error_tdoa != float('inf') else np.nan)
            
            # Mejor DOA
            min_error_doa = float('inf')
            for metodo, resultado in d['doa'].items():
                if resultado.get('valido', False):
                    if resultado['error_absoluto_deg'] < min_error_doa:
                        min_error_doa = resultado['error_absoluto_deg']
            mejor_doa.append(min_error_doa if min_error_doa != float('inf') else np.nan)
        
        ax3_twin = ax3.twinx()
        line1 = ax3.plot(param_values, mejor_tdoa, 'bo-', label='Mejor TDOA', linewidth=2)
        line2 = ax3_twin.plot(param_values, mejor_doa, 'ro-', label='Mejor DOA', linewidth=2)
        
        ax3.set_xlabel(parametro.replace('_', ' ').title())
        ax3.set_ylabel('Error TDOA (ms)', color='blue')
        ax3_twin.set_ylabel('Error DOA (°)', color='red')
        ax3.set_title('Mejor Rendimiento por Parámetro')
        
        # Combinar leyendas
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax3.legend(lines, labels, loc='upper left')
        
        ax3.grid(True, alpha=0.3)
        
        # Gráfico 4: Distribución de errores
        ax4 = axes[1, 1]
        todos_errores_doa = []
        for d in datos:
            for metodo, resultado in d['doa'].items():
                if resultado.get('valido', False):
                    todos_errores_doa.append(resultado['error_absoluto_deg'])
        
        if todos_errores_doa:
            ax4.hist(todos_errores_doa, bins=20, alpha=0.7, edgecolor='black')
            ax4.axvline(np.mean(todos_errores_doa), color='red', linestyle='--', 
                       label=f'Media: {np.mean(todos_errores_doa):.2f}°')
            ax4.axvline(np.median(todos_errores_doa), color='green', linestyle='--', 
                       label=f'Mediana: {np.median(todos_errores_doa):.2f}°')
        
        ax4.set_xlabel('Error DOA (°)')
        ax4.set_ylabel('Frecuencia')
        ax4.set_title('Distribución de Errores DOA')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def generar_reporte_completo(self, 
                                resultados_parametricos: Dict,
                                comparacion: Dict) -> str:
        """
        Genera reporte completo de evaluación
        """
        reporte = "=" * 60 + "\n"
        reporte += "REPORTE COMPLETO DE EVALUACIÓN DOA/TDOA\n"
        reporte += "=" * 60 + "\n\n"
        
        # Resumen ejecutivo
        reporte += "RESUMEN EJECUTIVO\n"
        reporte += "-" * 20 + "\n"
        
        if 'tdoa' in comparacion and comparacion['tdoa']:
            reporte += f"Mejor método TDOA: {comparacion['tdoa']['mejor_metodo']}\n"
            reporte += f"Error TDOA mínimo: {comparacion['tdoa']['mejor_error_ms']:.3f} ms\n"
            reporte += f"Mejora vs peor método: {comparacion['tdoa']['mejora_relativa']:.1f}%\n\n"
        
        if 'doa' in comparacion and comparacion['doa']:
            reporte += f"Mejor método DOA: {comparacion['doa']['mejor_metodo']}\n"
            reporte += f"Error DOA mínimo: {comparacion['doa']['mejor_error_deg']:.2f}°\n"
            reporte += f"Mejora vs peor método: {comparacion['doa']['mejora_relativa']:.1f}%\n\n"
        
        # Análisis por parámetro
        for param, datos in resultados_parametricos.items():
            if datos:
                reporte += f"ANÁLISIS VS {param.upper()}\n"
                reporte += "-" * 30 + "\n"
                
                # Estadísticas básicas
                errores_doa = []
                for d in datos:
                    for metodo, resultado in d['doa'].items():
                        if resultado.get('valido', False):
                            errores_doa.append(resultado['error_absoluto_deg'])
                
                if errores_doa:
                    reporte += f"Error DOA promedio: {np.mean(errores_doa):.2f}° ± {np.std(errores_doa):.2f}°\n"
                    reporte += f"Error DOA rango: [{np.min(errores_doa):.2f}°, {np.max(errores_doa):.2f}°]\n"
                    reporte += f"Número de evaluaciones: {len(errores_doa)}\n\n"
        
        # Recomendaciones
        reporte += "RECOMENDACIONES\n"
        reporte += "-" * 15 + "\n"
        
        if 'ranking_tdoa' in comparacion and comparacion['ranking_tdoa']:
            mejor_tdoa = comparacion['ranking_tdoa'][0]
            reporte += f"• Para TDOA, usar {mejor_tdoa['metodo']} (error: {mejor_tdoa['error_medio_ms']:.3f} ms)\n"
        
        if 'ranking_doa' in comparacion and comparacion['ranking_doa']:
            mejor_doa = comparacion['ranking_doa'][0]
            reporte += f"• Para DOA, usar {mejor_doa['metodo']} (error: {mejor_doa['error_absoluto_deg']:.2f}°)\n"
        
        reporte += "• Considerar trade-off entre precisión y robustez al ruido\n"
        reporte += "• Evaluar requisitos computacionales en aplicación real\n"
        
        return reporte
    
    def exportar_resultados(self, 
                           resultados: Dict, 
                           filename: str = "resultados_evaluacion.json"):
        """
        Exporta resultados a archivo JSON
        """
        # Convertir numpy arrays a listas para serialización
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        resultados_serializables = convert_numpy(resultados)
        
        with open(filename, 'w') as f:
            json.dump(resultados_serializables, f, indent=2)
        
        print(f"Resultados exportados a: {filename}")

# Ejemplo de uso y testing
if __name__ == "__main__":
    print("=== TESTING MÓDULO EVALUACIÓN ===")
    
    # Crear evaluador
    evaluador = EvaluadorDOA()
    
    # Test básico de cálculo de errores
    estimados = [29.5, 31.2, 28.8, 30.1]
    reales = [30.0, 30.0, 30.0, 30.0]
    
    error_abs = evaluador.calcular_error(estimados, reales, 'absoluto')
    error_ang = evaluador.calcular_error(estimados, reales, 'angular')
    
    print("Test de cálculo de errores:")
    print(f"Error absoluto medio: {error_abs['error_medio']:.3f}")
    print(f"Error angular medio: {error_ang['error_medio']:.3f}°")
    print(f"RMSE: {error_abs['rmse']:.3f}")
    
    # Simular comparación de métodos
    resultados_tdoa_sim = {
        'correlacion': {
            'error_medio_ms': 0.15,
            'error_std_ms': 0.05,
            'confidence_media': 0.7,
            'valido': True
        },
        'gcc_phat': {
            'error_medio_ms': 0.08,
            'error_std_ms': 0.03,
            'confidence_media': 0.9,
            'valido': True
        },
        'gcc_scot': {
            'error_medio_ms': 0.12,
            'error_std_ms': 0.04,
            'confidence_media': 0.8,
            'valido': True
        }
    }
    
    resultados_doa_sim = {
        'circular': {
            'error_absoluto_deg': 2.1,
            'std_estimacion_deg': 0.8,
            'num_estimaciones': 3,
            'valido': True
        },
        'ponderado': {
            'error_absoluto_deg': 1.8,
            'std_estimacion_deg': 0.6,
            'num_estimaciones': 3,
            'valido': True
        }
    }
    
    # Comparar algoritmos
    comparacion = evaluador.comparar_algoritmos(resultados_tdoa_sim, resultados_doa_sim)
    
    print(f"\nMejor método TDOA: {comparacion['tdoa']['mejor_metodo']}")
    print(f"Mejor método DOA: {comparacion['doa']['mejor_metodo']}")
    
    # Generar reporte
    reporte = evaluador.generar_reporte_completo({}, comparacion)
    print(f"\n{reporte}")
    
    print("\n¡Testing evaluación completado!")