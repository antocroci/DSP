#!/usr/bin/env python3
"""
Módulo de Simulación Acústica para Estimación DOA/TDOA
Utiliza pyroomacoustics para simular recintos y propagación acústica

Autor: Sistema DOA/TDOA
Fecha: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from mpl_toolkits.mplot3d import Axes3D
import pyroomacoustics as pra
from typing import Dict, List, Tuple, Optional, Union
import json
import os
from pathlib import Path
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime

# Configuración de warnings
warnings.filterwarnings('ignore', category=UserWarning)

@dataclass
class ConfiguracionRecinto:
    """
    Configuración completa del recinto acústico
    """
    # Dimensiones del recinto [largo, ancho, alto] en metros
    dimensiones: List[float] = None
    
    # Parámetros acústicos
    rt60: float = 0.3  # Tiempo de reverberación en segundos
    absorcion: float = None  # Coeficiente de absorción (calculado automáticamente si es None)
    
    # Tipo de recinto
    tipo: str = "reverberante"  # "anecoico" o "reverberante"
    
    # Parámetros de simulación
    fs: int = 16000  # Frecuencia de muestreo
    max_order: int = 3  # Orden máximo de reflexiones
    
    def __post_init__(self):
        if self.dimensiones is None:
            self.dimensiones = [10.0, 8.0, 3.0]  # Dimensiones por defecto
        
        # Calcular absorción si no se especifica
        if self.absorcion is None and self.tipo == "reverberante":
            # Fórmula de Sabine simplificada
            volumen = np.prod(self.dimensiones)
            superficie = 2 * (self.dimensiones[0] * self.dimensiones[1] + 
                            self.dimensiones[0] * self.dimensiones[2] + 
                            self.dimensiones[1] * self.dimensiones[2])
            self.absorcion = 0.161 * volumen / (self.rt60 * superficie)
            self.absorcion = np.clip(self.absorcion, 0.01, 0.99)

@dataclass
class ConfiguracionArray:
    """
    Configuración del array de micrófonos
    """
    # Número de micrófonos
    num_mics: int = 4
    
    # Separación entre micrófonos en metros
    spacing: float = 0.10
    
    # Posición del centro del array [x, y, z]
    posicion_centro: List[float] = None
    
    # Tipo de array: "lineal", "circular", "rectangular"
    tipo: str = "lineal"
    
    # Orientación del array en grados (solo para array lineal)
    orientacion: float = 0.0
    
    def __post_init__(self):
        if self.posicion_centro is None:
            self.posicion_centro = [2.0, 1.5, 1.0]  # Centro por defecto

@dataclass
class ConfiguracionFuente:
    """
    Configuración de la fuente sonora
    """
    # Posición de la fuente [x, y, z] o None para calcular desde ángulo/distancia
    posicion: List[float] = None
    
    # Parámetros polares (usados si posición es None)
    azimuth: float = 30.0  # Ángulo azimutal en grados
    elevation: float = 0.0  # Ángulo de elevación en grados
    distancia: float = 2.0  # Distancia en metros
    
    # Parámetros de la señal
    tipo_senal: str = "tono"  # "tono", "chirp", "ruido", "speech"
    frecuencia: float = 1000.0  # Frecuencia para tono en Hz
    duracion: float = 1.0  # Duración en segundos
    
    # Parámetros de ruido
    snr_db: float = 20.0  # Relación señal-ruido en dB
    agregar_ruido: bool = True

@dataclass
class ConfiguracionSimulacion:
    """
    Configuración completa de la simulación
    """
    recinto: ConfiguracionRecinto = None
    array: ConfiguracionArray = None
    fuentes: List[ConfiguracionFuente] = None
    
    # Metadatos
    nombre: str = "simulacion_default"
    descripcion: str = ""
    
    def __post_init__(self):
        if self.recinto is None:
            self.recinto = ConfiguracionRecinto()
        if self.array is None:
            self.array = ConfiguracionArray()
        if self.fuentes is None:
            self.fuentes = [ConfiguracionFuente()]

class SimuladorAcustico:
    """
    Simulador acústico principal usando pyroomacoustics
    """
    
    def __init__(self, config: ConfiguracionSimulacion = None):
        """
        Inicializa el simulador
        
        Args:
            config: Configuración de la simulación
        """
        self.config = config if config is not None else ConfiguracionSimulacion()
        self.room = None
        self.signals = {}
        self.metadata = {}
        self.resultados = {}
        
        # Validar configuración
        self._validar_configuracion()
        
    def _validar_configuracion(self):
        """
        Valida la configuración de la simulación
        """
        # Validar dimensiones del recinto
        if len(self.config.recinto.dimensiones) != 3:
            raise ValueError("Las dimensiones del recinto deben ser [largo, ancho, alto]")
        
        if any(d <= 0 for d in self.config.recinto.dimensiones):
            raise ValueError("Todas las dimensiones deben ser positivas")
        
        # Validar parámetros acústicos
        if self.config.recinto.rt60 <= 0:
            raise ValueError("RT60 debe ser positivo")
        
        if self.config.recinto.absorcion is not None:
            if not 0 < self.config.recinto.absorcion < 1:
                raise ValueError("La absorción debe estar entre 0 y 1")
        
        # Validar array
        if self.config.array.num_mics < 2:
            raise ValueError("Se necesitan al menos 2 micrófonos")
        
        if self.config.array.spacing <= 0:
            raise ValueError("La separación entre micrófonos debe ser positiva")
        
        # Validar que el array cabe en el recinto
        array_size = (self.config.array.num_mics - 1) * self.config.array.spacing
        if array_size >= min(self.config.recinto.dimensiones[:2]):
            warnings.warn("El array podría ser demasiado grande para el recinto")
    
    def crear_recinto(self):
        """
        Crea el recinto acústico usando pyroomacoustics
        """
        print(f"Creando recinto {self.config.recinto.tipo}...")
        
        if self.config.recinto.tipo == "anecoico":
            # Recinto anecoico (sin reflexiones)
            self.room = pra.ShoeBox(
                self.config.recinto.dimensiones,
                fs=self.config.recinto.fs,
                max_order=0,  # Sin reflexiones
                absorption=1.0  # Absorción total
            )
            print(f"  Recinto anecoico: {self.config.recinto.dimensiones}")
            
        elif self.config.recinto.tipo == "reverberante":
            # Recinto reverberante
            self.room = pra.ShoeBox(
                self.config.recinto.dimensiones,
                fs=self.config.recinto.fs,
                max_order=self.config.recinto.max_order,
                absorption=self.config.recinto.absorcion
            )
            print(f"  Recinto reverberante: {self.config.recinto.dimensiones}")
            print(f"  RT60: {self.config.recinto.rt60:.2f}s, Absorción: {self.config.recinto.absorcion:.3f}")
            
        else:
            raise ValueError(f"Tipo de recinto desconocido: {self.config.recinto.tipo}")
        
        # Guardar metadatos del recinto
        self.metadata['recinto'] = {
            'dimensiones': self.config.recinto.dimensiones,
            'volumen': np.prod(self.config.recinto.dimensiones),
            'tipo': self.config.recinto.tipo,
            'rt60_objetivo': self.config.recinto.rt60,
            'absorcion': self.config.recinto.absorcion,
            'fs': self.config.recinto.fs,
            'max_order': self.config.recinto.max_order
        }
    
    def crear_array_microfonos(self):
        """
        Crea el array de micrófonos
        """
        if self.room is None:
            raise RuntimeError("Debe crear el recinto antes del array de micrófonos")
        
        print(f"Creando array de {self.config.array.num_mics} micrófonos...")
        
        # Calcular posiciones de los micrófonos
        posiciones = self._calcular_posiciones_microfonos()
        
        # Agregar micrófonos al recinto
        for i, pos in enumerate(posiciones):
            self.room.add_microphone(pos)
            print(f"  Mic {i+1}: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
        
        # Guardar metadatos del array
        self.metadata['array'] = {
            'num_mics': self.config.array.num_mics,
            'spacing': self.config.array.spacing,
            'tipo': self.config.array.tipo,
            'posicion_centro': self.config.array.posicion_centro,
            'posiciones': posiciones.tolist(),
            'orientacion': self.config.array.orientacion
        }
    
    def _calcular_posiciones_microfonos(self) -> np.ndarray:
        """
        Calcula las posiciones de los micrófonos según la configuración
        """
        centro = np.array(self.config.array.posicion_centro)
        num_mics = self.config.array.num_mics
        spacing = self.config.array.spacing
        
        if self.config.array.tipo == "lineal":
            # Array lineal
            # Calcular posiciones relativas al centro
            start_offset = -(num_mics - 1) * spacing / 2
            offsets = np.arange(num_mics) * spacing + start_offset
            
            # Aplicar orientación
            angle_rad = np.deg2rad(self.config.array.orientacion)
            dx = offsets * np.cos(angle_rad)
            dy = offsets * np.sin(angle_rad)
            dz = np.zeros(num_mics)
            
            posiciones = centro[:, np.newaxis] + np.array([dx, dy, dz])
            
        elif self.config.array.tipo == "circular":
            # Array circular
            radio = spacing * num_mics / (2 * np.pi)
            angulos = np.linspace(0, 2*np.pi, num_mics, endpoint=False)
            
            dx = radio * np.cos(angulos)
            dy = radio * np.sin(angulos)
            dz = np.zeros(num_mics)
            
            posiciones = centro[:, np.newaxis] + np.array([dx, dy, dz])
            
        elif self.config.array.tipo == "rectangular":
            # Array rectangular (2x2, 3x2, etc.)
            filas = int(np.sqrt(num_mics))
            cols = int(np.ceil(num_mics / filas))
            
            posiciones = []
            for i in range(num_mics):
                fila = i // cols
                col = i % cols
                
                x = centro[0] + (col - (cols-1)/2) * spacing
                y = centro[1] + (fila - (filas-1)/2) * spacing
                z = centro[2]
                
                posiciones.append([x, y, z])
            
            posiciones = np.array(posiciones).T
            
        else:
            raise ValueError(f"Tipo de array desconocido: {self.config.array.tipo}")
        
        # Validar que todas las posiciones están dentro del recinto
        for i, pos in enumerate(posiciones.T):
            if not self._posicion_valida(pos):
                raise ValueError(f"Micrófono {i+1} fuera del recinto: {pos}")
        
        return posiciones
    
    def agregar_fuentes(self):
        """
        Agrega todas las fuentes sonoras configuradas
        """
        if self.room is None:
            raise RuntimeError("Debe crear el recinto antes de agregar fuentes")
        
        print(f"Agregando {len(self.config.fuentes)} fuente(s)...")
        
        self.metadata['fuentes'] = []
        
        for i, config_fuente in enumerate(self.config.fuentes):
            # Calcular posición de la fuente
            if config_fuente.posicion is not None:
                posicion = np.array(config_fuente.posicion)
            else:
                posicion = self._calcular_posicion_fuente(config_fuente)
            
            # Validar posición
            if not self._posicion_valida(posicion):
                raise ValueError(f"Fuente {i+1} fuera del recinto: {posicion}")
            
            # Crear señal
            signal = self._crear_senal(config_fuente)
            
            # Agregar fuente al recinto
            self.room.add_source(posicion, signal=signal)
            
            print(f"  Fuente {i+1}: [{posicion[0]:.2f}, {posicion[1]:.2f}, {posicion[2]:.2f}]")
            print(f"    Tipo: {config_fuente.tipo_senal}, Duración: {config_fuente.duracion:.2f}s")
            
            # Guardar metadatos
            self.metadata['fuentes'].append({
                'posicion': posicion.tolist(),
                'azimuth': config_fuente.azimuth,
                'elevation': config_fuente.elevation,
                'distancia': config_fuente.distancia,
                'tipo_senal': config_fuente.tipo_senal,
                'frecuencia': config_fuente.frecuencia,
                'duracion': config_fuente.duracion,
                'snr_db': config_fuente.snr_db
            })
    
    def _calcular_posicion_fuente(self, config_fuente: ConfiguracionFuente) -> np.ndarray:
        """
        Calcula la posición de una fuente a partir de coordenadas polares
        """
        centro_array = np.array(self.config.array.posicion_centro)
        
        # Convertir a radianes
        azimuth_rad = np.deg2rad(config_fuente.azimuth)
        elevation_rad = np.deg2rad(config_fuente.elevation)
        
        # Calcular posición cartesiana
        x = centro_array[0] + config_fuente.distancia * np.cos(elevation_rad) * np.cos(azimuth_rad)
        y = centro_array[1] + config_fuente.distancia * np.cos(elevation_rad) * np.sin(azimuth_rad)
        z = centro_array[2] + config_fuente.distancia * np.sin(elevation_rad)
        
        return np.array([x, y, z])
    
    def _crear_senal(self, config_fuente: ConfiguracionFuente) -> np.ndarray:
        """
        Crea la señal de audio según la configuración
        """
        fs = self.config.recinto.fs
        duracion = config_fuente.duracion
        t = np.linspace(0, duracion, int(duracion * fs), endpoint=False)
        
        if config_fuente.tipo_senal == "tono":
            signal = np.sin(2 * np.pi * config_fuente.frecuencia * t)
            
        elif config_fuente.tipo_senal == "chirp":
            f0 = 500  # Frecuencia inicial
            f1 = 2000  # Frecuencia final
            signal = np.sin(2 * np.pi * (f0 + (f1 - f0) * t / duracion) * t)
            
        elif config_fuente.tipo_senal == "ruido":
            signal = np.random.randn(len(t))
            
        elif config_fuente.tipo_senal == "speech":
            # Simulación de voz (múltiples tonos)
            freqs = [200, 400, 800, 1600]  # Formantes típicos
            signal = np.zeros(len(t))
            for freq in freqs:
                signal += np.sin(2 * np.pi * freq * t) / len(freqs)
            
        else:
            raise ValueError(f"Tipo de señal desconocido: {config_fuente.tipo_senal}")
        
        # Normalizar
        signal = signal / np.max(np.abs(signal))
        
        return signal
    
    def _posicion_valida(self, posicion: np.ndarray) -> bool:
        """
        Verifica si una posición está dentro del recinto
        """
        dims = self.config.recinto.dimensiones
        return (0 < posicion[0] < dims[0] and 
                0 < posicion[1] < dims[1] and 
                0 < posicion[2] < dims[2])
    
    def simular_propagacion(self):
        """
        Ejecuta la simulación de propagación acústica
        """
        if self.room is None:
            raise RuntimeError("Debe configurar el recinto antes de simular")
        
        if len(self.room.sources) == 0:
            raise RuntimeError("Debe agregar al menos una fuente antes de simular")
        
        if self.room.mic_array is None:
            raise RuntimeError("Debe agregar micrófonos antes de simular")
        
        print("Ejecutando simulación de propagación...")
        
        # Ejecutar simulación
        self.room.simulate()
        
        # Obtener señales de los micrófonos
        mic_signals = self.room.mic_array.signals
        
        # Agregar ruido si se solicita
        for i, config_fuente in enumerate(self.config.fuentes):
            if config_fuente.agregar_ruido:
                mic_signals = self._agregar_ruido(mic_signals, config_fuente.snr_db)
        
        # Guardar señales
        self.signals = {
            'mic_signals': mic_signals,
            'source_signals': [source.signal for source in self.room.sources],
            'fs': self.config.recinto.fs,
            'duracion': mic_signals.shape[1] / self.config.recinto.fs
        }
        
        print(f"  Simulación completada: {mic_signals.shape[0]} mics, {mic_signals.shape[1]} muestras")
        print(f"  Duración: {self.signals['duracion']:.2f}s")
        
        # Calcular métricas de la simulación
        self._calcular_metricas_simulacion()
    
    def _agregar_ruido(self, signals: np.ndarray, snr_db: float) -> np.ndarray:
        """
        Agrega ruido gaussiano a las señales
        """
        # Calcular potencia de la señal
        signal_power = np.mean(signals**2, axis=1, keepdims=True)
        
        # Calcular potencia del ruido necesaria
        snr_linear = 10**(snr_db / 10)
        noise_power = signal_power / snr_linear
        
        # Generar ruido
        noise = np.random.randn(*signals.shape) * np.sqrt(noise_power)
        
        return signals + noise
    
    def _calcular_metricas_simulacion(self):
        """
        Calcula métricas de calidad de la simulación
        """
        mic_signals = self.signals['mic_signals']
        
        # SNR real de cada micrófono
        snr_real = []
        for i in range(mic_signals.shape[0]):
            signal_power = np.var(mic_signals[i, :])
            # Estimar ruido de los últimos 10% de la señal (asumiendo silencio)
            noise_samples = mic_signals[i, -int(0.1 * mic_signals.shape[1]):]
            noise_power = np.var(noise_samples)
            
            if noise_power > 0:
                snr = 10 * np.log10(signal_power / noise_power)
            else:
                snr = float('inf')
            
            snr_real.append(snr)
        
        # RT60 real (si es reverberante)
        rt60_real = None
        if self.config.recinto.tipo == "reverberante":
            rt60_real = self._estimar_rt60()
        
        self.metadata['metricas'] = {
            'snr_real_db': snr_real,
            'snr_promedio_db': np.mean(snr_real),
            'rt60_real': rt60_real,
            'nivel_maximo_db': 20 * np.log10(np.max(np.abs(mic_signals))),
            'duracion_real': self.signals['duracion']
        }
    
    def _estimar_rt60(self) -> float:
        """
        Estima el RT60 real de la simulación
        """
        try:
            # Usar la primera señal de micrófono
            signal = self.signals['mic_signals'][0, :]
            
            # Calcular envolvente de energía
            energy = signal**2
            
            # Suavizar con ventana móvil
            window_size = int(0.01 * self.config.recinto.fs)  # 10ms
            energy_smooth = np.convolve(energy, np.ones(window_size)/window_size, mode='same')
            
            # Convertir a dB
            energy_db = 10 * np.log10(energy_smooth + 1e-10)
            
            # Encontrar el pico
            peak_idx = np.argmax(energy_db)
            
            # Buscar punto -60dB
            peak_level = energy_db[peak_idx]
            target_level = peak_level - 60
            
            # Encontrar cuando cruza -60dB
            decay_part = energy_db[peak_idx:]
            cross_idx = np.where(decay_part < target_level)[0]
            
            if len(cross_idx) > 0:
                rt60_samples = cross_idx[0]
                rt60_seconds = rt60_samples / self.config.recinto.fs
                return rt60_seconds
            else:
                return None
                
        except Exception:
            return None
    
    def visualizar_setup(self, mostrar_trayectorias: bool = False):
        """
        Visualiza la configuración del recinto, micrófonos y fuentes
        """
        if self.room is None:
            raise RuntimeError("Debe crear el recinto antes de visualizar")
        
        fig = plt.figure(figsize=(15, 5))
        
        # Vista 2D (planta)
        ax1 = fig.add_subplot(131)
        self._plot_vista_planta(ax1)
        
        # Vista 3D
        ax2 = fig.add_subplot(132, projection='3d')
        self._plot_vista_3d(ax2)
        
        # Información del setup
        ax3 = fig.add_subplot(133)
        self._plot_info_setup(ax3)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_vista_planta(self, ax):
        """
        Dibuja vista en planta (2D)
        """
        dims = self.config.recinto.dimensiones
        
        # Dibujar recinto
        rect = Rectangle((0, 0), dims[0], dims[1], 
                        linewidth=2, edgecolor='black', facecolor='lightgray', alpha=0.3)
        ax.add_patch(rect)
        
        # Dibujar micrófonos
        if 'array' in self.metadata:
            posiciones = np.array(self.metadata['array']['posiciones'])
            ax.scatter(posiciones[0, :], posiciones[1, :], 
                      c='blue', s=100, marker='s', label='Micrófonos', zorder=5)
            
            # Numerar micrófonos
            for i, (x, y) in enumerate(posiciones[:2, :].T):
                ax.annotate(f'M{i+1}', (x, y), xytext=(5, 5), 
                           textcoords='offset points', fontsize=8)
        
        # Dibujar fuentes
        if 'fuentes' in self.metadata:
            for i, fuente in enumerate(self.metadata['fuentes']):
                pos = fuente['posicion']
                ax.scatter(pos[0], pos[1], c='red', s=150, marker='*', 
                          label=f'Fuente {i+1}' if i == 0 else '', zorder=5)
                ax.annotate(f'S{i+1}', (pos[0], pos[1]), xytext=(5, 5), 
                           textcoords='offset points', fontsize=8)
        
        ax.set_xlim(-0.5, dims[0] + 0.5)
        ax.set_ylim(-0.5, dims[1] + 0.5)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Vista en Planta')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    def _plot_vista_3d(self, ax):
        """
        Dibuja vista 3D
        """
        dims = self.config.recinto.dimensiones
        
        # Dibujar esquinas del recinto
        corners = np.array([
            [0, 0, 0], [dims[0], 0, 0], [dims[0], dims[1], 0], [0, dims[1], 0],
            [0, 0, dims[2]], [dims[0], 0, dims[2]], [dims[0], dims[1], dims[2]], [0, dims[1], dims[2]]
        ])
        
        # Dibujar aristas del recinto
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Base
            [4, 5], [5, 6], [6, 7], [7, 4],  # Techo
            [0, 4], [1, 5], [2, 6], [3, 7]   # Verticales
        ]
        
        for edge in edges:
            points = corners[edge]
            ax.plot3D(*points.T, 'k-', alpha=0.3)
        
        # Dibujar micrófonos
        if 'array' in self.metadata:
            posiciones = np.array(self.metadata['array']['posiciones'])
            ax.scatter(posiciones[0, :], posiciones[1, :], posiciones[2, :], 
                      c='blue', s=100, marker='s', label='Micrófonos')
        
        # Dibujar fuentes
        if 'fuentes' in self.metadata:
            for i, fuente in enumerate(self.metadata['fuentes']):
                pos = fuente['posicion']
                ax.scatter(pos[0], pos[1], pos[2], c='red', s=150, marker='*', 
                          label=f'Fuente {i+1}' if i == 0 else '')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Vista 3D')
        ax.legend()
    
    def _plot_info_setup(self, ax):
        """
        Muestra información del setup
        """
        ax.axis('off')
        
        info_text = []
        info_text.append("CONFIGURACIÓN DE LA SIMULACIÓN")
        info_text.append("=" * 30)
        
        # Información del recinto
        info_text.append(f"Recinto: {self.config.recinto.tipo}")
        info_text.append(f"Dimensiones: {self.config.recinto.dimensiones} m")
        info_text.append(f"Volumen: {np.prod(self.config.recinto.dimensiones):.1f} m³")
        
        if self.config.recinto.tipo == "reverberante":
            info_text.append(f"RT60: {self.config.recinto.rt60:.2f} s")
            info_text.append(f"Absorción: {self.config.recinto.absorcion:.3f}")
        
        info_text.append("")
        
        # Información del array
        info_text.append(f"Array: {self.config.array.num_mics} micrófonos")
        info_text.append(f"Tipo: {self.config.array.tipo}")
        info_text.append(f"Separación: {self.config.array.spacing*100:.1f} cm")
        
        info_text.append("")
        
        # Información de fuentes
        info_text.append(f"Fuentes: {len(self.config.fuentes)}")
        for i, fuente in enumerate(self.config.fuentes):
            info_text.append(f"  S{i+1}: {fuente.tipo_senal}, {fuente.azimuth:.0f}°, {fuente.distancia:.1f}m")
        
        # Mostrar métricas si están disponibles
        if 'metricas' in self.metadata:
            info_text.append("")
            info_text.append("MÉTRICAS DE SIMULACIÓN")
            info_text.append("-" * 20)
            metricas = self.metadata['metricas']
            info_text.append(f"SNR promedio: {metricas['snr_promedio_db']:.1f} dB")
            if metricas['rt60_real'] is not None:
                info_text.append(f"RT60 real: {metricas['rt60_real']:.2f} s")
        
        # Mostrar texto
        y_pos = 0.95
        for line in info_text:
            ax.text(0.05, y_pos, line, transform=ax.transAxes, 
                   fontsize=9, verticalalignment='top',
                   fontweight='bold' if line.isupper() else 'normal')
            y_pos -= 0.05
    
    def guardar_configuracion(self, filename: str):
        """
        Guarda la configuración en un archivo JSON
        """
        config_dict = {
            'recinto': asdict(self.config.recinto),
            'array': asdict(self.config.array),
            'fuentes': [asdict(fuente) for fuente in self.config.fuentes],
            'nombre': self.config.nombre,
            'descripcion': self.config.descripcion,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"Configuración guardada en: {filename}")
    
    def cargar_configuracion(self, filename: str):
        """
        Carga configuración desde un archivo JSON
        """
        with open(filename, 'r') as f:
            config_dict = json.load(f)
        
        # Reconstruir configuración
        recinto = ConfiguracionRecinto(**config_dict['recinto'])
        array = ConfiguracionArray(**config_dict['array'])
        fuentes = [ConfiguracionFuente(**f) for f in config_dict['fuentes']]
        
        self.config = ConfiguracionSimulacion(
            recinto=recinto,
            array=array,
            fuentes=fuentes,
            nombre=config_dict.get('nombre', 'cargado'),
            descripcion=config_dict.get('descripcion', '')
        )
        
        print(f"Configuración cargada desde: {filename}")
    
    def exportar_resultados(self, directorio: str = "resultados"):
        """
        Exporta todos los resultados de la simulación
        """
        # Crear directorio si no existe
        Path(directorio).mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{self.config.nombre}_{timestamp}"
        
        # Guardar configuración
        config_file = Path(directorio) / f"{base_name}_config.json"
        self.guardar_configuracion(config_file)
        
        # Guardar metadatos
        metadata_file = Path(directorio) / f"{base_name}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        # Guardar señales (formato NPZ)
        if self.signals:
            signals_file = Path(directorio) / f"{base_name}_signals.npz"
            np.savez_compressed(signals_file, **self.signals)
        
        print(f"Resultados exportados en directorio: {directorio}")
        print(f"  - Configuración: {config_file.name}")
        print(f"  - Metadatos: {metadata_file.name}")
        if self.signals:
            print(f"  - Señales: {signals_file.name}")
    
    def ejecutar_simulacion_completa(self):
        """
        Ejecuta una simulación completa desde la configuración
        """
        print(f"=== SIMULACIÓN: {self.config.nombre} ===")
        
        # Crear recinto
        self.crear_recinto()
        
        # Crear array de micrófonos
        self.crear_array_microfonos()
        
        # Agregar fuentes
        self.agregar_fuentes()
        
        # Simular propagación
        self.simular_propagacion()
        
        print("✓ Simulación completa finalizada")
        
        return self.signals, self.metadata

# Funciones de utilidad para crear configuraciones comunes

def crear_config_campo_libre(azimuth: float = 30.0, distancia: float = 2.0) -> ConfiguracionSimulacion:
    """
    Crea configuración para simulación en campo libre (anecoico)
    """
    recinto = ConfiguracionRecinto(
        dimensiones=[20.0, 15.0, 8.0],  # Recinto grande
        tipo="anecoico"
    )
    
    array = ConfiguracionArray(
        num_mics=4,
        spacing=0.10,
        posicion_centro=[10.0, 7.5, 4.0]  # Centro del recinto
    )
    
    fuente = ConfiguracionFuente(
        azimuth=azimuth,
        distancia=distancia,
        tipo_senal="tono",
        frecuencia=1000.0,
        snr_db=30.0  # SNR alto para campo libre
    )
    
    return ConfiguracionSimulacion(
        recinto=recinto,
        array=array,
        fuentes=[fuente],
        nombre="campo_libre",
        descripcion=f"Simulación en campo libre, azimuth={azimuth}°, distancia={distancia}m"
    )

def crear_config_reverberante(rt60: float = 0.5, azimuth: float = 30.0) -> ConfiguracionSimulacion:
    """
    Crea configuración para simulación reverberante
    """
    recinto = ConfiguracionRecinto(
        dimensiones=[8.0, 6.0, 3.0],
        tipo="reverberante",
        rt60=rt60
    )
    
    array = ConfiguracionArray(
        num_mics=4,
        spacing=0.10,
        posicion_centro=[4.0, 3.0, 1.5]
    )
    
    fuente = ConfiguracionFuente(
        azimuth=azimuth,
        distancia=2.0,
        tipo_senal="speech",
        snr_db=15.0  # SNR más bajo para ambiente reverberante
    )
    
    return ConfiguracionSimulacion(
        recinto=recinto,
        array=array,
        fuentes=[fuente],
        nombre="reverberante",
        descripcion=f"Simulación reverberante, RT60={rt60}s, azimuth={azimuth}°"
    )

# Ejemplo de uso y testing
if __name__ == "__main__":
    print("=== TESTING MÓDULO SIMULACIÓN ===")
    
    # Test 1: Simulación en campo libre
    print("\n1. Simulación en campo libre...")
    config_libre = crear_config_campo_libre(azimuth=45.0, distancia=3.0)
    sim_libre = SimuladorAcustico(config_libre)
    
    try:
        signals_libre, metadata_libre = sim_libre.ejecutar_simulacion_completa()
        print(f"✓ Campo libre exitoso: {signals_libre['mic_signals'].shape}")
        
        # Visualizar
        sim_libre.visualizar_setup()
        
    except Exception as e:
        print(f"❌ Error en campo libre: {e}")
    
    # Test 2: Simulación reverberante
    print("\n2. Simulación reverberante...")
    config_rev = crear_config_reverberante(rt60=0.8, azimuth=60.0)
    sim_rev = SimuladorAcustico(config_rev)
    
    try:
        signals_rev, metadata_rev = sim_rev.ejecutar_simulacion_completa()
        print(f"✓ Reverberante exitoso: {signals_rev['mic_signals'].shape}")
        
        # Comparar métricas
        print(f"RT60 objetivo: {config_rev.recinto.rt60:.2f}s")
        if metadata_rev['metricas']['rt60_real']:
            print(f"RT60 real: {metadata_rev['metricas']['rt60_real']:.2f}s")
        
    except Exception as e:
        print(f"❌ Error en reverberante: {e}")
    
    # Test 3: Guardar y cargar configuración
    print("\n3. Test de configuración...")
    try:
        sim_libre.guardar_configuracion("test_config.json")
        
        sim_test = SimuladorAcustico()
        sim_test.cargar_configuracion("test_config.json")
        print("✓ Configuración guardada y cargada correctamente")
        
        # Limpiar archivo de test
        os.remove("test_config.json")
        
    except Exception as e:
        print(f"❌ Error en configuración: {e}")
    
    print("\n¡Testing simulación completado!")
