import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, ifft, fftfreq
from typing import Tuple, List, Dict, Optional
import warnings

class EstimadorTDOA:
    """
    Clase para estimación de TDOA (Time Difference of Arrival) y DOA (Direction of Arrival)
    """
    
    def __init__(self, fs: int = 16000, c: float = 343.0):
        """
        Inicializa el estimador TDOA
        
        Args:
            fs: Frecuencia de muestreo en Hz
            c: Velocidad del sonido en m/s
        """
        self.fs = fs
        self.c = c
        self.dt = 1.0 / fs  # Período de muestreo
        
    def correlacion_cruzada(self, 
                           x1: np.ndarray, 
                           x2: np.ndarray, 
                           metodo: str = 'full') -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula la correlación cruzada clásica entre dos señales
        
        Args:
            x1, x2: Señales de entrada
            metodo: 'full', 'valid', o 'same'
            
        Returns:
            correlation: Correlación cruzada
            lags: Vector de retardos correspondientes
        """
        # Asegurar que las señales tengan la misma longitud
        min_len = min(len(x1), len(x2))
        x1 = x1[:min_len]
        x2 = x2[:min_len]
        
        # Calcular correlación cruzada
        correlation = np.correlate(x1, x2, mode=metodo)
        
        # Calcular vector de retardos
        if metodo == 'full':
            lags = np.arange(-len(x2) + 1, len(x1))
        elif metodo == 'same':
            lags = np.arange(-len(x2)//2, len(x1)//2)
        else:  # valid
            lags = np.arange(len(x1) - len(x2) + 1)
            
        return correlation, lags
    
    def gcc_basico(self, 
                   x1: np.ndarray, 
                   x2: np.ndarray, 
                   ventana: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generalized Cross-Correlation (GCC) básico
        
        Args:
            x1, x2: Señales de entrada
            ventana: Tipo de ventana ('hann', 'hamming', etc.)
            
        Returns:
            gcc: Correlación cruzada generalizada
            lags: Vector de retardos
        """
        # Asegurar misma longitud
        min_len = min(len(x1), len(x2))
        x1 = x1[:min_len]
        x2 = x2[:min_len]
        
        # Aplicar ventana si se especifica
        if ventana:
            window = signal.get_window(ventana, min_len)
            x1 = x1 * window
            x2 = x2 * window
        
        # Calcular FFT
        X1 = fft(x1, n=2*min_len-1)
        X2 = fft(x2, n=2*min_len-1)
        
        # Cross-power spectrum
        cross_spectrum = X1 * np.conj(X2)
        
        # IFFT para obtener correlación
        gcc = np.real(ifft(cross_spectrum))
        gcc = np.fft.fftshift(gcc)
        
        # Vector de retardos
        lags = np.arange(-min_len + 1, min_len)
        
        return gcc, lags
    
    def gcc_phat(self, 
                 x1: np.ndarray, 
                 x2: np.ndarray, 
                 epsilon: float = 1e-12) -> Tuple[np.ndarray, np.ndarray]:
        """
        GCC-PHAT (Phase Transform) - Método más robusto para TDOA
        
        Args:
            x1, x2: Señales de entrada
            epsilon: Pequeño valor para evitar división por cero
            
        Returns:
            gcc_phat: Correlación GCC-PHAT
            lags: Vector de retardos
        """
        # Asegurar misma longitud
        min_len = min(len(x1), len(x2))
        x1 = x1[:min_len]
        x2 = x2[:min_len]
        
        # Calcular FFT con zero-padding para mejor resolución
        n_fft = 2 * min_len - 1
        X1 = fft(x1, n=n_fft)
        X2 = fft(x2, n=n_fft)
        
        # Cross-power spectrum
        cross_spectrum = X1 * np.conj(X2)
        
        # Phase Transform (PHAT) - Normalización por magnitud
        magnitude = np.abs(cross_spectrum) + epsilon
        phat_spectrum = cross_spectrum / magnitude
        
        # IFFT para obtener correlación
        gcc_phat = np.real(ifft(phat_spectrum))
        gcc_phat = np.fft.fftshift(gcc_phat)
        
        # Vector de retardos en muestras
        lags = np.arange(-min_len + 1, min_len)
        
        return gcc_phat, lags
    
    def gcc_scot(self, 
                 x1: np.ndarray, 
                 x2: np.ndarray, 
                 epsilon: float = 1e-12) -> Tuple[np.ndarray, np.ndarray]:
        """
        GCC-SCOT (Smoothed Coherence Transform)
        
        Args:
            x1, x2: Señales de entrada
            epsilon: Pequeño valor para evitar división por cero
            
        Returns:
            gcc_scot: Correlación GCC-SCOT
            lags: Vector de retardos
        """
        min_len = min(len(x1), len(x2))
        x1 = x1[:min_len]
        x2 = x2[:min_len]
        
        n_fft = 2 * min_len - 1
        X1 = fft(x1, n=n_fft)
        X2 = fft(x2, n=n_fft)
        
        # Auto-power spectra
        S11 = X1 * np.conj(X1)
        S22 = X2 * np.conj(X2)
        
        # Cross-power spectrum
        S12 = X1 * np.conj(X2)
        
        # SCOT weighting
        scot_weight = 1.0 / (np.sqrt(S11 * S22) + epsilon)
        scot_spectrum = S12 * scot_weight
        
        # IFFT
        gcc_scot = np.real(ifft(scot_spectrum))
        gcc_scot = np.fft.fftshift(gcc_scot)
        
        lags = np.arange(-min_len + 1, min_len)
        
        return gcc_scot, lags
    
    def estimar_tdoa_par(self, 
                        mic1: np.ndarray, 
                        mic2: np.ndarray, 
                        metodo: str = 'gcc_phat',
                        busqueda_pico: str = 'max',
                        ventana_busqueda: Optional[Tuple[float, float]] = None) -> Dict:
        """
        Estima TDOA entre un par de micrófonos
        
        Args:
            mic1, mic2: Señales de los micrófonos
            metodo: 'correlacion', 'gcc', 'gcc_phat', 'gcc_scot'
            busqueda_pico: 'max', 'interpolacion'
            ventana_busqueda: (min_delay, max_delay) en segundos
            
        Returns:
            Diccionario con resultados de TDOA
        """
        # Seleccionar método de correlación
        if metodo == 'correlacion':
            correlation, lags = self.correlacion_cruzada(mic1, mic2)
        elif metodo == 'gcc':
            correlation, lags = self.gcc_basico(mic1, mic2)
        elif metodo == 'gcc_phat':
            correlation, lags = self.gcc_phat(mic1, mic2)
        elif metodo == 'gcc_scot':
            correlation, lags = self.gcc_scot(mic1, mic2)
        else:
            raise ValueError(f"Método desconocido: {metodo}")
        
        # Aplicar ventana de búsqueda si se especifica
        if ventana_busqueda:
            min_samples = int(ventana_busqueda[0] * self.fs)
            max_samples = int(ventana_busqueda[1] * self.fs)
            
            # Encontrar índices válidos
            valid_indices = (lags >= min_samples) & (lags <= max_samples)
            if np.any(valid_indices):
                correlation = correlation[valid_indices]
                lags = lags[valid_indices]
        
        # Encontrar pico máximo
        if busqueda_pico == 'max':
            max_idx = np.argmax(np.abs(correlation))
            tdoa_samples = lags[max_idx]
            confidence = np.abs(correlation[max_idx])
            
        elif busqueda_pico == 'interpolacion':
            # Interpolación parabólica para sub-sample precision
            max_idx = np.argmax(np.abs(correlation))
            
            if 0 < max_idx < len(correlation) - 1:
                # Interpolación parabólica
                y1, y2, y3 = np.abs(correlation[max_idx-1:max_idx+2])
                a = (y1 - 2*y2 + y3) / 2
                b = (y3 - y1) / 2
                
                if a != 0:
                    offset = -b / (2*a)
                    tdoa_samples = lags[max_idx] + offset
                else:
                    tdoa_samples = lags[max_idx]
            else:
                tdoa_samples = lags[max_idx]
                
            confidence = np.abs(correlation[max_idx])
        
        # Convertir a tiempo
        tdoa_seconds = tdoa_samples * self.dt
        
        # Calcular métricas adicionales
        correlation_normalized = correlation / np.max(np.abs(correlation))
        
        return {
            'tdoa_samples': float(tdoa_samples),
            'tdoa_seconds': float(tdoa_seconds),
            'confidence': float(confidence),
            'correlation': correlation,
            'lags': lags,
            'correlation_normalized': correlation_normalized,
            'metodo': metodo,
            'max_idx': int(max_idx) if 'max_idx' in locals() else None
        }
    
    def estimar_tdoa_array(self, 
                          signals: np.ndarray, 
                          referencia: int = 0,
                          metodo: str = 'gcc_phat') -> Dict:
        """
        Estima TDOA para todo el array usando un micrófono de referencia
        
        Args:
            signals: Matriz de señales (num_mics x num_samples)
            referencia: Índice del micrófono de referencia
            metodo: Método de correlación a usar
            
        Returns:
            Diccionario con TDOAs de todos los pares
        """
        num_mics = signals.shape[0]
        resultados = {}
        
        ref_signal = signals[referencia, :]
        
        for i in range(num_mics):
            if i != referencia:
                par_key = f"mic_{referencia+1}_mic_{i+1}"
                resultado = self.estimar_tdoa_par(ref_signal, signals[i, :], metodo=metodo)
                resultados[par_key] = resultado
        
        return resultados
    
    def calcular_doa_lineal(self, 
                           tdoas: Dict, 
                           spacing: float,
                           array_geometry: str = 'linear') -> Dict:
        """
        Calcula DOA para array lineal usando TDOAs
        
        Args:
            tdoas: Diccionario con TDOAs estimados
            spacing: Separación entre micrófonos en metros
            array_geometry: Tipo de geometría del array
            
        Returns:
            Diccionario con ángulos estimados
        """
        angulos = {}
        
        if array_geometry == 'linear':
            for par_key, tdoa_data in tdoas.items():
                tdoa_sec = tdoa_data['tdoa_seconds']
                
                # Calcular ángulo usando geometría lineal
                # sin(θ) = c * TDOA / d
                sin_theta = (self.c * tdoa_sec) / spacing
                
                # Verificar que esté en rango válido
                if abs(sin_theta) <= 1.0:
                    theta_rad = np.arcsin(sin_theta)
                    theta_deg = np.degrees(theta_rad)
                    
                    angulos[par_key] = {
                        'angulo_rad': float(theta_rad),
                        'angulo_deg': float(theta_deg),
                        'sin_theta': float(sin_theta),
                        'tdoa_usado': float(tdoa_sec),
                        'valido': True
                    }
                else:
                    angulos[par_key] = {
                        'angulo_rad': None,
                        'angulo_deg': None,
                        'sin_theta': float(sin_theta),
                        'tdoa_usado': float(tdoa_sec),
                        'valido': False,
                        'error': 'TDOA fuera de rango físico'
                    }
        
        return angulos
    
    def evaluar_error_angular(self, 
                             angulo_estimado: float, 
                             angulo_real: float) -> Dict:
        """
        Evalúa el error en la estimación angular
        
        Args:
            angulo_estimado: Ángulo estimado en grados
            angulo_real: Ángulo real en grados
            
        Returns:
            Diccionario con métricas de error
        """
        error_absoluto = abs(angulo_estimado - angulo_real)
        error_relativo = error_absoluto / abs(angulo_real) * 100 if angulo_real != 0 else float('inf')
        
        return {
            'error_absoluto_deg': float(error_absoluto),
            'error_relativo_pct': float(error_relativo),
            'angulo_estimado': float(angulo_estimado),
            'angulo_real': float(angulo_real)
        }
    
    def visualizar_correlacion(self, 
                              resultado_tdoa: Dict, 
                              titulo: str = "Correlación Cruzada"):
        """
        Visualiza el resultado de correlación cruzada
        """
        correlation = resultado_tdoa['correlation']
        lags = resultado_tdoa['lags']
        tdoa_samples = resultado_tdoa['tdoa_samples']
        
        plt.figure(figsize=(12, 6))
        
        # Subplot 1: Correlación completa
        plt.subplot(1, 2, 1)
        plt.plot(lags * self.dt * 1000, correlation)  # Convertir a ms
        plt.axvline(x=tdoa_samples * self.dt * 1000, color='red', linestyle='--', 
                   label=f'TDOA = {tdoa_samples * self.dt * 1000:.2f} ms')
        plt.xlabel('Retardo (ms)')
        plt.ylabel('Correlación')
        plt.title(f'{titulo} - Vista Completa')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Subplot 2: Zoom alrededor del pico
        plt.subplot(1, 2, 2)
        max_idx = resultado_tdoa.get('max_idx', np.argmax(np.abs(correlation)))
        window = 50  # muestras alrededor del pico
        start_idx = max(0, max_idx - window)
        end_idx = min(len(correlation), max_idx + window)
        
        plt.plot(lags[start_idx:end_idx] * self.dt * 1000, 
                correlation[start_idx:end_idx])
        plt.axvline(x=tdoa_samples * self.dt * 1000, color='red', linestyle='--',
                   label=f'TDOA = {tdoa_samples * self.dt * 1000:.2f} ms')
        plt.xlabel('Retardo (ms)')
        plt.ylabel('Correlación')
        plt.title(f'{titulo} - Zoom en Pico')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def comparar_metodos(self, 
                        mic1: np.ndarray, 
                        mic2: np.ndarray,
                        metodos: List[str] = ['correlacion', 'gcc_phat', 'gcc_scot']) -> Dict:
        """
        Compara diferentes métodos de estimación TDOA
        """
        resultados = {}
        
        for metodo in metodos:
            try:
                resultado = self.estimar_tdoa_par(mic1, mic2, metodo=metodo)
                resultados[metodo] = resultado
            except Exception as e:
                print(f"Error con método {metodo}: {e}")
                resultados[metodo] = None
        
        return resultados
    
    def analisis_rendimiento(self, 
                           signals: np.ndarray,
                           angulos_reales: List[float],
                           distancias_reales: List[float],
                           spacing: float) -> Dict:
        """
        Análisis completo de rendimiento del sistema TDOA/DOA
        """
        # Estimar TDOAs
        tdoas = self.estimar_tdoa_array(signals)
        
        # Calcular DOAs
        doas = self.calcular_doa_lineal(tdoas, spacing)
        
        # Evaluar errores
        errores = {}
        for i, (par_key, doa_data) in enumerate(doas.items()):
            if doa_data['valido'] and i < len(angulos_reales):
                error = self.evaluar_error_angular(
                    doa_data['angulo_deg'], 
                    angulos_reales[i]
                )
                errores[par_key] = error
        
        return {
            'tdoas': tdoas,
            'doas': doas,
            'errores': errores,
            'resumen': {
                'num_estimaciones': len(doas),
                'estimaciones_validas': sum(1 for d in doas.values() if d['valido']),
                'error_promedio': np.mean([e['error_absoluto_deg'] for e in errores.values()]) if errores else None
            }
        }

# Funciones de utilidad
def crear_senal_test(tipo: str = "chirp", duracion: float = 1.0, fs: int = 16000) -> np.ndarray:
    """
    Crea señales de prueba para testing
    """
    t = np.linspace(0, duracion, int(duracion * fs))
    
    if tipo == "chirp":
        return signal.chirp(t, f0=500, f1=2000, t1=duracion, method='linear')
    elif tipo == "tono":
        return np.sin(2 * np.pi * 1000 * t)
    elif tipo == "ruido":
        return np.random.randn(len(t))
    elif tipo == "impulso":
        impulso = np.zeros(len(t))
        impulso[len(t)//4] = 1.0
        return impulso
    else:
        return signal.chirp(t, f0=500, f1=2000, t1=duracion, method='linear')

# Ejemplo de uso y testing
if __name__ == "__main__":
    print("=== TESTING MÓDULO TDOA ===")
    
    # Crear estimador
    estimador = EstimadorTDOA(fs=16000)
    
    # Crear señales de prueba con TDOA conocido
    fs = 16000
    duracion = 1.0
    tdoa_real = 0.001  # 1 ms de retardo
    
    # Señal original
    signal_orig = crear_senal_test("chirp", duracion, fs)
    
    # Señal retardada
    delay_samples = int(tdoa_real * fs)
    signal_delayed = np.zeros_like(signal_orig)
    signal_delayed[delay_samples:] = signal_orig[:-delay_samples]
    
    # Agregar ruido
    snr_db = 20
    noise_power = np.var(signal_orig) / (10**(snr_db/10))
    signal_orig += np.sqrt(noise_power) * np.random.randn(len(signal_orig))
    signal_delayed += np.sqrt(noise_power) * np.random.randn(len(signal_delayed))
    
    print(f"TDOA real: {tdoa_real*1000:.2f} ms ({delay_samples} muestras)")
    
    # Comparar métodos
    metodos = ['correlacion', 'gcc_phat', 'gcc_scot']
    resultados = estimador.comparar_metodos(signal_orig, signal_delayed, metodos)
    
    print("\n=== RESULTADOS POR MÉTODO ===")
    for metodo, resultado in resultados.items():
        if resultado:
            tdoa_est = resultado['tdoa_seconds']
            error = abs(tdoa_est - tdoa_real) * 1000  # en ms
            print(f"{metodo:12}: TDOA = {tdoa_est*1000:6.2f} ms, Error = {error:.2f} ms")
    
    # Visualizar mejor resultado (GCC-PHAT)
    if resultados['gcc_phat']:
        estimador.visualizar_correlacion(resultados['gcc_phat'], "GCC-PHAT")
    
    print("\n¡Testing completado!")
