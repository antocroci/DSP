import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os
from typing import Tuple, List, Optional, Dict
import json

class SimuladorDOA:
    """
    Simulador para estimación de dirección de arribo (DOA) usando pyroomacoustics
    """
    
    def __init__(self, fs: int = 16000):
        """
        Inicializa el simulador
        
        Args:
            fs: Frecuencia de muestreo en Hz
        """
        self.fs = fs
        self.c = 343  # Velocidad del sonido en m/s
        self.array_geometry = None
        self.room = None
        self.signals = {}
        self.source_positions = []
        
        # Nuevos parámetros para ruido ambiente
        self.ruido_ambiente_config = {
            'habilitado': False,
            'nivel_db': -40.0,
            'tipo': 'rosa'  # 'blanco', 'rosa', 'marron'
        }
        
    def crear_array_microfonos(self, 
                                num_mics: int = 4, 
                                spacing: float = 0.1,
                                center_pos: List[float] = [2.0, 1.5, 1.0]) -> np.ndarray:
        """
        Crea un array lineal de micrófonos
        
        Args:
            num_mics: Número de micrófonos (default: 4)
            spacing: Separación entre micrófonos en metros (default: 0.1m = 10cm)
            center_pos: Posición del centro del array [x, y, z]
            
        Returns:
            Array con las posiciones de los micrófonos (3 x num_mics)
        """
        print("\n2. ARRAY DE MICRÓFONOS:")
        print("   Posición del centro del array:")
    
        center = np.array([float(input("     Posición X [default: 2.0]: ") or 2), float(input("     Posición Y [default: 1.5]: ") or 1.5), float(input("     Posición Z [default: 1.0]: ") or 1.0)]) 
        
        
        spacing = float(input("Ingrese el espacio entre micrófonos (default 0.1): ") or 0.1)
        
        mic_positions = np.zeros((3, num_mics))
        
        # Calcular posiciones simétricas alrededor del centro
        start_offset = -(num_mics - 1) * spacing / 2
        
        for i in range(num_mics):
            mic_positions[0, i] = center[0] + start_offset + i * spacing  # X
            mic_positions[1, i] = center[1]  # Y
            mic_positions[2, i] = center[2]  # Z
            
        self.array_geometry = {
            'num_mics': num_mics,
            'positions': mic_positions,
            'spacing': spacing,
            'center': center
        }
        
        print(f"Array creado: {num_mics} micrófonos, separación {spacing*100:.1f}cm")
        for i in range(num_mics):
            print(f"  Mic {i+1}: [{mic_positions[0,i]:.2f}, {mic_positions[1,i]:.2f}, {mic_positions[2,i]:.2f}]")
            
        return mic_positions
    
    def simular_ambiente_anecoico(self, 
                                room_size: List[float] = [10, 8, 3],
                                max_order = 0,
                                absorption = 1,
                                air_absorption=False) -> object:
        """
        Crea un ambiente anecoico (sin reflexiones)
        """
        import pyroomacoustics as pra
        
        self.room = pra.ShoeBox(
            room_size,
            fs=self.fs,
            max_order= max_order,  # Sin reflexiones
            absorption= absorption,
            air_absorption= air_absorption
        )
        
        if self.array_geometry is None:
            self.crear_array_microfonos()
            
        self.room.add_microphone_array(self.array_geometry['positions'])
        
        print(f"Ambiente anecoico creado: {room_size[0]}x{room_size[1]}x{room_size[2]}m")
        return self.room
    
    def simular_ambiente_reverberante(self, 
                                    room_size: List[float] = [6, 4, 3],
                                    rt60: float = 0.3,
                                    ) -> object:
        """
        Crea un ambiente reverberante con opción de ruido ambiente
        
        Args:
            room_size: Dimensiones del recinto [x, y, z] en metros
            rt60: Tiempo de reverberación en segundos
            ruido_ambiente: Si agregar ruido ambiente del recinto
            nivel_ruido_db: Nivel de ruido ambiente en dB (relativo al pico de señal)
            tipo_ruido: Tipo de ruido ambiente ('blanco', 'rosa', 'marron')
        """
        import pyroomacoustics as pra
        
        # Calcular absorción para RT60 deseado
        volume = np.prod(room_size)
        surface_area = 2 * (room_size[0]*room_size[1] + 
                            room_size[0]*room_size[2] + 
                            room_size[1]*room_size[2])
        
        absorption = 0.161 * volume / (rt60 * surface_area)
        absorption = min(absorption, 0.99)
        
        self.room = pra.ShoeBox(
            room_size,
            fs=self.fs,
            max_order=10,
            absorption=absorption,
            air_absorption=True
        )
        
        if self.array_geometry is None:
            self.crear_array_microfonos()
            
        self.room.add_microphone_array(self.array_geometry['positions'])
        
        # Configurar ruido ambiente

        print("\n3. RUIDO AMBIENTE:")
        ruido_habilitado = input("   ¿Agregar ruido ambiente? (s/n) [default: n]: ").lower() or "n"
        self.ruido_ambiente_config['habilitado'] = (ruido_habilitado == 's')
        
        if self.ruido_ambiente_config['habilitado']:
            self.ruido_ambiente_config['nivel_db'] = float(input("   Nivel de ruido ambiente en dB [default: -40]: ") or -40)
            print("   Tipos de ruido: 1=Blanco, 2=Rosa, 3=Marrón")
            tipo_ruido_num = input("   Tipo de ruido [default: 2]: ") or 2
            tipos = {'1': 'blanco', '2': 'rosa', '3': 'marron'}
            self.ruido_ambiente_config['tipo'] = tipos.get(tipo_ruido_num, 'rosa')
    
        
        print(f"Ambiente reverberante creado: RT60={rt60:.2f}s, Absorción={absorption:.3f}")
        
        if self.ruido_ambiente_config['habilitado']:
            print(f"  Ruido ambiente: {self.ruido_ambiente_config['tipo']}, {self.ruido_ambiente_config['nivel_db']:.1f} dB")

        
        return self.room
    
    def agregar_fuente(self, 
                        signal: np.ndarray,
                        azimuth: float,
                        distance: float = 2.0,
                        elevation: float = 0.0) -> List[float]:
        """
        Agrega una fuente sonora al ambiente
        """
        if self.room is None:
            raise ValueError("Debe crear un ambiente primero")
            
        # Convertir ángulos a radianes
        azimuth_rad = np.deg2rad(azimuth)
        elevation_rad = np.deg2rad(elevation)
        
        # Calcular posición de la fuente
        center = self.array_geometry['center']
        
        source_pos = [
            float(center[0] + distance * np.cos(elevation_rad) * np.cos(azimuth_rad)),
            float(center[1] + distance * np.cos(elevation_rad) * np.sin(azimuth_rad)),
            float(center[2] + distance * np.sin(elevation_rad))
        ]
        
        # Agregar fuente al recinto
        self.room.add_source(source_pos, signal=signal)
        
        # Guardar información (ya como lista, no numpy array)
        self.source_positions.append({
            'position': source_pos,
            'azimuth': float(azimuth),
            'elevation': float(elevation),
            'distance': float(distance)
        })
        
        print(f"Fuente agregada en [{source_pos[0]:.2f}, {source_pos[1]:.2f}, {source_pos[2]:.2f}]")
        print(f"  Azimuth: {azimuth:.1f}°, Distancia: {distance:.2f}m")
        
        return source_pos
    
    def _generar_ruido_rosa(self, num_samples: int) -> np.ndarray:
        """
        Genera ruido rosa usando el algoritmo de Voss-McCartney
        """
        # Número de generadores
        num_sources = 16
        
        # Inicializar generadores
        values = np.random.randn(num_sources)
        max_key = 0x1f  # 31 en binario
        
        noise = np.zeros(num_samples)
        
        for i in range(num_samples):
            # Determinar qué generadores actualizar
            key = i & max_key
            last_key = (i - 1) & max_key
            diff = last_key ^ key
            
            # Actualizar generadores que cambiaron
            for j in range(num_sources):
                if diff & (1 << j):
                    values[j] = np.random.randn()
            
            # Sumar todos los generadores
            noise[i] = np.sum(values)
        
        return noise
    
    def _generar_ruido_marron(self, num_samples: int) -> np.ndarray:
        """
        Genera ruido marrón (Browniano) integrando ruido blanco
        """
        white_noise = np.random.randn(num_samples)
        brown_noise = np.cumsum(white_noise)
        
        # Normalizar para evitar deriva
        brown_noise = brown_noise - np.mean(brown_noise)
        
        return brown_noise
    
    def _generar_ruido_coloreado(self, shape: Tuple, tipo_ruido: str, noise_power: float) -> np.ndarray:
        """
        Genera diferentes tipos de ruido coloreado
        """
        num_mics, num_samples = shape
        
        if tipo_ruido == "blanco":
            # Ruido blanco (espectro plano)
            noise = np.random.randn(num_mics, num_samples)
            
        elif tipo_ruido == "rosa":
            # Ruido rosa (1/f, -3dB/octava)
            noise = np.zeros((num_mics, num_samples))
            for mic in range(num_mics):
                noise[mic, :] = self._generar_ruido_rosa(num_samples)
        
        elif tipo_ruido == "marron":
            # Ruido marrón (1/f², -6dB/octava)
            noise = np.zeros((num_mics, num_samples))
            for mic in range(num_mics):
                noise[mic, :] = self._generar_ruido_marron(num_samples)
        
        else:
            raise ValueError(f"Tipo de ruido desconocido: {tipo_ruido}")
        
        # Normalizar y escalar a la potencia deseada
        for mic in range(num_mics):
            current_power = np.var(noise[mic, :])
            if current_power > 0:
                noise[mic, :] = noise[mic, :] * np.sqrt(noise_power / current_power)
        
        return noise
    
    def _agregar_ruido_ambiente(self, signals: np.ndarray) -> np.ndarray:
        """
        Agrega ruido ambiente del recinto
        """
        if not self.ruido_ambiente_config['habilitado']:
            return signals
        
        # Calcular potencia máxima de la señal
        max_signal_power = np.max(np.mean(signals**2, axis=1))
        
        # Calcular potencia del ruido ambiente
        nivel_ruido_linear = 10**(self.ruido_ambiente_config['nivel_db'] / 10)
        noise_power = max_signal_power * nivel_ruido_linear
        
        # Generar ruido según el tipo especificado
        noise = self._generar_ruido_coloreado(
            signals.shape, 
            self.ruido_ambiente_config['tipo'], 
            noise_power
        )
        
        return signals + noise
    
    def simular_propagacion(self, agregar_ruido: bool = False, snr_db: float = 20):
        """
        Simula la propagación acústica
        """
        if self.room is None:
            raise ValueError("Debe crear un ambiente y agregar fuentes primero")
            
        self.room.simulate()
        mic_signals = self.room.mic_array.signals
        
        # Agregar ruido ambiente del recinto si está habilitado
        mic_signals = self._agregar_ruido_ambiente(mic_signals)
        
        # Agregar ruido adicional de grabación si se solicita
        if agregar_ruido:
            for i in range(mic_signals.shape[0]):
                signal_power = np.mean(mic_signals[i, :] ** 2)
                noise_power = signal_power / (10 ** (snr_db / 10))
                noise = np.sqrt(noise_power) * np.random.randn(mic_signals.shape[1])
                mic_signals[i, :] += noise
        
        self.signals = {
            'mic_signals': mic_signals,
            'fs': self.fs,
            'num_mics': mic_signals.shape[0],
            'length': mic_signals.shape[1],
            'snr_db': snr_db if agregar_ruido else None,
            'ruido_ambiente': self.ruido_ambiente_config
        }
        
        print(f"Simulación completada: {mic_signals.shape[0]} mics, {mic_signals.shape[1]} muestras")
        if agregar_ruido:
            print(f"  SNR adicional: {snr_db:.1f} dB")
        if self.ruido_ambiente_config['habilitado']:
            print(f"  Ruido ambiente: {self.ruido_ambiente_config['tipo']}, {self.ruido_ambiente_config['nivel_db']:.1f} dB")
    
    def guardar_senales(self, directorio: str = "simulaciones", 
                        nombre_experimento: str = "exp_001"):
        """
        Guarda las señales simuladas y metadatos
        """
        if not self.signals:
            raise ValueError("No hay señales para guardar")
            
        os.makedirs(directorio, exist_ok=True)
        
        # Guardar señales individuales
        for i in range(self.signals['num_mics']):
            filename = os.path.join(directorio, f"{nombre_experimento}_mic_{i+1}.wav")
            signal_norm = self.signals['mic_signals'][i, :] / np.max(np.abs(self.signals['mic_signals']))
            signal_int16 = (signal_norm * 32767).astype(np.int16)
            wavfile.write(filename, self.fs, signal_int16)
        
        # Preparar metadatos (convertir todo a tipos serializables)
        metadata = {
            'experimento': nombre_experimento,
            'fs': int(self.fs),
            'num_mics': int(self.signals['num_mics']),
            'array_geometry': {
                'num_mics': int(self.array_geometry['num_mics']),
                'spacing': float(self.array_geometry['spacing']),
                'center': [float(x) for x in self.array_geometry['center']],
                'positions': [[float(x) for x in col] for col in self.array_geometry['positions'].T]
            },
            'room_info': {
                'shoebox_dim': [float(x) for x in self.room.shoebox_dim],
                'max_order': int(self.room.max_order)
            },
            'sources': self.source_positions,  # Ya están como tipos básicos
            'snr_db': float(self.signals['snr_db']) if self.signals.get('snr_db') is not None else None,
            'ruido_ambiente': {
                'habilitado': bool(self.signals['ruido_ambiente']['habilitado']),
                'nivel_db': float(self.signals['ruido_ambiente']['nivel_db']),
                'tipo': str(self.signals['ruido_ambiente']['tipo'])
            },
            'length_samples': int(self.signals['length']),
            'duration_s': float(self.signals['length'] / self.fs)
        }
        
        metadata_file = os.path.join(directorio, f"{nombre_experimento}_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Archivos guardados en: {directorio}/")
        print(f"  {self.signals['num_mics']} archivos WAV + metadatos JSON")
    
    def cargar_senales(self, directorio: str, nombre_experimento: str) -> Dict:
        """
        Carga señales previamente guardadas
        
        Args:
            directorio: Directorio donde están los archivos
            nombre_experimento: Nombre del experimento
            
        Returns:
            Diccionario con señales y metadatos
        """
        # Cargar metadatos
        metadata_file = os.path.join(directorio, f"{nombre_experimento}_metadata.json")
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Cargar señales
        num_mics = metadata['num_mics']
        signals = []
        
        for i in range(num_mics):
            filename = os.path.join(directorio, f"{nombre_experimento}_mic_{i+1}.wav")
            fs, signal = wavfile.read(filename)
            signals.append(signal.astype(np.float32) / 32767.0)  # Normalizar
        
        signals_array = np.array(signals)
        
        result = {
            'mic_signals': signals_array,
            'metadata': metadata,
            'fs': fs
        }
        
        print(f"Señales cargadas: {nombre_experimento}")
        print(f"  {num_mics} micrófonos, {signals_array.shape[1]} muestras")
        if 'ruido_ambiente' in metadata and metadata['ruido_ambiente']['habilitado']:
            print(f"  Ruido ambiente: {metadata['ruido_ambiente']['tipo']}, {metadata['ruido_ambiente']['nivel_db']:.1f} dB")
        
        return result
    
    def visualizar_setup(self):
        """
        Visualiza la configuración del experimento
        """
        if self.room is None:
            print("No hay ambiente creado para visualizar")
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Vista superior (XY)
        room_corners = np.array([
            [0, 0], [self.room.shoebox_dim[0], 0],
            [self.room.shoebox_dim[0], self.room.shoebox_dim[1]],
            [0, self.room.shoebox_dim[1]], [0, 0]
        ])
        ax1.plot(room_corners[:, 0], room_corners[:, 1], 'k-', linewidth=2, label='Recinto')
        
        # Micrófonos
        mic_pos = self.array_geometry['positions']
        ax1.scatter(mic_pos[0, :], mic_pos[1, :], c='blue', s=100, marker='o', label='Micrófonos')
        
        for i in range(mic_pos.shape[1]):
            ax1.annotate(f'M{i+1}', (mic_pos[0, i], mic_pos[1, i]), 
                        xytext=(5, 5), textcoords='offset points')
        
        # Fuentes
        for i, source_info in enumerate(self.source_positions):
            pos = source_info['position']
            ax1.scatter(pos[0], pos[1], c='red', s=150, marker='*', 
                        label='Fuentes' if i == 0 else "")
            ax1.annotate(f'S{i+1}\n{source_info["azimuth"]:.0f}°', 
                        (pos[0], pos[1]), xytext=(5, 5), textcoords='offset points')
        
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_title('Vista Superior')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # Vista polar
        ax2 = plt.subplot(122, projection='polar')
        for i, source_info in enumerate(self.source_positions):
            azimuth_rad = np.deg2rad(source_info['azimuth'])
            distance = source_info['distance']
            ax2.scatter(azimuth_rad, distance, c='red', s=150, marker='*')
            ax2.annotate(f'S{i+1}', (azimuth_rad, distance))
        
        ax2.set_title('Vista Polar')
        ax2.grid(True)
        
        # Agregar información de ruido ambiente en el título
        if self.ruido_ambiente_config['habilitado']:
            fig.suptitle(f'Configuración del Experimento - Ruido Ambiente: {self.ruido_ambiente_config["tipo"]} ({self.ruido_ambiente_config["nivel_db"]:.1f} dB)')
        else:
            fig.suptitle('Configuración del Experimento')
        
        plt.tight_layout()
        plt.show()
        
    def calcular_angulos_individuales(self):
        """
        Calcula los ángulos de incidencia de cada fuente con respecto a cada micrófono individualmente.
        También calcula las diferencias de distancia entre cada micrófono y la fuente con respecto al micrófono 1.

        Returns:
            dict: Diccionario con los ángulos azimutales individuales y diferencias de distancia para cada fuente.
        """
        if self.room is None or not self.source_positions:
            print("No hay datos para calcular ángulos individuales.")
            return {}

        mic_positions = self.array_geometry['positions']
        num_mic = mic_positions.shape[1]

        resultados = {}
        for s_idx, source_info in enumerate(self.source_positions):
            pos = source_info['position']
            angulos_individuales = []
            distancias = []

            for m_idx in range(num_mic):
                # Calcular el vector entre el micrófono y la fuente
                dx = pos[0] - mic_positions[0, m_idx]
                dy = pos[1] - mic_positions[1, m_idx]

                # Calcular ángulo azimutal
                angulo_azimutal = np.degrees(np.arctan2(dy, dx))
                angulos_individuales.append(angulo_azimutal)

                # Calcular la distancia entre el micrófono y la fuente
                distancia = np.sqrt(dx**2 + dy**2)
                distancias.append(distancia)

            # Calcular diferencias de distancia con respecto al micrófono 1
            diferencias_distancia = [distancias[i] - distancias[0] for i in range(num_mic)]

            resultados[f'fuente_{s_idx+1}'] = {
                'angulos_individuales': angulos_individuales,
                'distancias': distancias,
                'diferencias_distancia': diferencias_distancia
            }

        return resultados
    
    def visualizar_geometria_detallada(self):
        """
        Visualización detallada mostrando ángulos individuales
        """
        if self.room is None or not self.source_positions:
            print("No hay datos para visualizar")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        mic_pos = self.array_geometry['positions']
        colors = ['blue', 'green', 'orange', 'purple']
        
        # Vista superior con líneas desde cada micrófono
        ax1.set_title('Vista Superior - Líneas de Vista por Micrófono')
        
        # Dibujar recinto
        room_corners = np.array([
            [0, 0], [self.room.shoebox_dim[0], 0],
            [self.room.shoebox_dim[0], self.room.shoebox_dim[1]],
            [0, self.room.shoebox_dim[1]], [0, 0]
        ])
        ax1.plot(room_corners[:, 0], room_corners[:, 1], 'k-', linewidth=2)
        
        # Dibujar micrófonos
        for i in range(mic_pos.shape[1]):
            ax1.scatter(mic_pos[0, i], mic_pos[1, i], c=colors[i % len(colors)], s=100, 
                        marker='o', label=f'Mic {i+1}')
            ax1.annotate(f'M{i+1}', (mic_pos[0, i], mic_pos[1, i]), 
                        xytext=(5, 5), textcoords='offset points')
        
        # Dibujar fuentes y líneas desde cada micrófono
        for s_idx, source_info in enumerate(self.source_positions):
            pos = source_info['position']
            ax1.scatter(pos[0], pos[1], c='red', s=200, marker='*')
            ax1.annotate(f'S{s_idx+1}', (pos[0], pos[1]), 
                        xytext=(5, 5), textcoords='offset points')
            
            # Líneas desde cada micrófono a la fuente
            for i in range(mic_pos.shape[1]):
                ax1.plot([mic_pos[0, i], pos[0]], [mic_pos[1, i], pos[1]], 
                        color=colors[i % len(colors)], linestyle='--', alpha=0.7, linewidth=1)
        
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # Gráfico de ángulos por micrófono
        ax2.set_title('Ángulos Azimutales por Micrófono')
        resultados = self.calcular_angulos_individuales()
        
        for s_idx, (fuente_key, datos) in enumerate(resultados.items()):
            mic_nums = list(range(1, len(datos['angulos_individuales']) + 1))
            ax2.plot(mic_nums, datos['angulos_individuales'], 'o-', 
                    label=f'Fuente {s_idx+1}', linewidth=2, markersize=8)
        
        ax2.set_xlabel('Número de Micrófono')
        ax2.set_ylabel('Ángulo Azimutal (°)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        # Set x-axis ticks to integers only
        import matplotlib.ticker as ticker
        ax2.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        
        # Diferencias de distancia (importante para TDOA)
        ax3.set_title('Diferencias de Distancia vs Mic 1')
        for s_idx, (fuente_key, datos) in enumerate(resultados.items()):
            mic_nums = list(range(1, len(datos['diferencias_distancia']) + 1))
            ax3.plot(mic_nums, datos['diferencias_distancia'], 'o-', 
                    label=f'Fuente {s_idx+1}', linewidth=2, markersize=8)
        
        ax3.set_xlabel('Número de Micrófono')
        ax3.set_ylabel('Diferencia de Distancia (m)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        # TDOA esperados
        ax4.set_title('TDOA Esperados vs Mic 1')
        for s_idx, (fuente_key, datos) in enumerate(resultados.items()):
            mic_nums = list(range(1, len(datos['diferencias_distancia']) + 1))
            tdoas = [d / self.c * 1e6 for d in datos['diferencias_distancia']]  # en μs
            ax4.plot(mic_nums, tdoas, 'o-', 
                    label=f'Fuente {s_idx+1}', linewidth=2, markersize=8)
        
        ax4.set_xlabel('Número de Micrófono')
        ax4.set_ylabel('TDOA (μs)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        # Agregar información de ruido en el título general
        if self.ruido_ambiente_config['habilitado']:
            fig.suptitle(f'Análisis Geométrico Detallado - Ruido Ambiente: {self.ruido_ambiente_config["tipo"]} ({self.ruido_ambiente_config["nivel_db"]:.1f} dB)')
        else:
            fig.suptitle('Análisis Geométrico Detallado')
        
        plt.tight_layout()
        plt.show()

    def analizar_espectro_ruido(self):
        """
        Analiza el espectro del ruido ambiente generado
        """
        if not self.signals or not self.ruido_ambiente_config['habilitado']:
            print("No hay señales con ruido ambiente para analizar")
            return
        
        # Tomar una muestra del final de la señal (donde debería haber más ruido)
        mic_signals = self.signals['mic_signals']
        noise_samples = mic_signals[:, -int(0.2 * mic_signals.shape[1]):]  # Últimos 20%
        
        # Calcular FFT para el primer micrófono
        fft_noise = np.fft.fft(noise_samples[0, :])
        freqs = np.fft.fftfreq(len(fft_noise), 1/self.fs)
        
        # Solo frecuencias positivas
        positive_freqs = freqs[:len(freqs)//2]
        magnitude = np.abs(fft_noise[:len(fft_noise)//2])
        
        # Convertir a dB
        magnitude_db = 20 * np.log10(magnitude + 1e-10)
        
        # Graficar
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(positive_freqs, magnitude_db)
        plt.xlabel('Frecuencia (Hz)')
        plt.ylabel('Magnitud (dB)')
        plt.title(f'Espectro del Ruido Ambiente - {self.ruido_ambiente_config["tipo"].title()}')
        plt.grid(True, alpha=0.3)
        plt.xlim(0, self.fs//2)
        
        plt.subplot(1, 2, 2)
        plt.loglog(positive_freqs[1:], magnitude[1:])  # Evitar f=0
        plt.xlabel('Frecuencia (Hz)')
        plt.ylabel('Magnitud')
        plt.title('Espectro en Escala Log-Log')
        plt.grid(True, alpha=0.3)
        
        # Agregar líneas de referencia teóricas
        if self.ruido_ambiente_config['tipo'] == 'rosa':
            # Ruido rosa: -3dB/octava = -10dB/década
            ref_line = magnitude[10] * (positive_freqs[1:] / positive_freqs[10])**(-0.5)
            plt.loglog(positive_freqs[1:], ref_line, 'r--', alpha=0.7, label='Teórico -3dB/oct')
        elif self.ruido_ambiente_config['tipo'] == 'marron':
            # Ruido marrón: -6dB/octava = -20dB/década
            ref_line = magnitude[10] * (positive_freqs[1:] / positive_freqs[10])**(-1.0)
            plt.loglog(positive_freqs[1:], ref_line, 'r--', alpha=0.7, label='Teórico -6dB/oct')
        
        plt.legend()
        plt.tight_layout()
        plt.show()

# Función para crear señales de prueba
def crear_senal_prueba(tipo: str = "tono", duracion: float = 2.0, fs: int = 16000) -> np.ndarray:
    """
    Crea diferentes tipos de señales de prueba
    """
    t = np.linspace(0, duracion, int(duracion * fs))
    
    if tipo == "tono":
        return np.sin(2 * np.pi * 1000 * t)  # 1kHz
    elif tipo == "chirp":
        return np.sin(2 * np.pi * (500 + 1000 * t / duracion) * t)  # 500Hz a 1.5kHz
    elif tipo == "ruido":
        return np.random.randn(len(t))
    else:
        return np.sin(2 * np.pi * 1000 * t)

# Ejemplo de uso
if __name__ == "__main__":
    print("=== SIMULADOR DOA CON RUIDO AMBIENTE - PRUEBA ===")

    print("\n2. PARÁMETROS ACÚSTICOS:")
    ambiente_tipo = int(input("   Tipo de ambiente (1=Anecoico, 2=Reverberante) [default: 2]: ") or 2)

    # Crear simulador
    sim = SimuladorDOA(fs=16000)

    if ambiente_tipo == 1:
        # Ambiente anecoico CON ruido ambiente
        sim.simular_ambiente_anecoico(
            room_size=[float(input("   Ancho del recinto (X) en metros [default: 6.0]: ") or 6.0), 
                        float(input("   Largo del recinto (Y) en metros [default: 4.0]: ") or 4.0),
                        float(input("   Altura del recinto (Z) en metros [default: 3.0]: ") or 3.0)], 
                        max_order = 0,
                        absorption = float(input("Ingrese el coeficiente de absorción de la sala (default 1): ") or 1),
                        air_absorption=False
    )
    else:
        # Ambiente reverberante CON ruido ambiente
        sim.simular_ambiente_reverberante(
            room_size=[float(input("   Ancho del recinto (X) en metros [default: 6.0]: ") or 6.0), 
                        float(input("   Largo del recinto (Y) en metros [default: 4.0]: ") or 4.0),
                        float(input("   Altura del recinto (Z) en metros [default: 3.0]: ") or 3.0)], 
            rt60= float(input("   RT60 en segundos [default: 0.3]: ") or 0.3),
    )
    
    # Crear señal de prueba
    signal = crear_senal_prueba("tono", duracion=1.0)

    # Configuración de fuentes
    print("\n5. FUENTES SONORAS:")
    sim.agregar_fuente(signal, azimuth=float(input(f"     Azimuth en grados [default: {60}]: ") or 60.0), distance=float(input("     Distancia en metros [default: 2.0]: ") or 2.0), elevation=float(input("     Elevación en grados [default: 0.0]: ") or 0.0))
    
    # Guardar y visualizar
    ### sim.guardar_senales(nombre_experimento="test_30deg_ruido_rosa")
    sim.visualizar_setup()
    
    # Visualizar geometría detallada
    sim.visualizar_geometria_detallada()
    
    print("\n¡Todas las simulaciones completadas exitosamente!")