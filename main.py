"""
Punto de entrada principal del proyecto MPEA.
Archivo: main.py

Ejecuta el servidor web de la interfaz y abre el navegador automáticamente
en http://127.0.0.1:5000 para interactuar con los modelos predictivos
del Módulo de Young y Límite Elástico.
"""

import sys
from pathlib import Path

# Configurar rutas del proyecto
DIRECTORIO_RAIZ = Path(__file__).resolve().parent
CARPETA_FRONT = DIRECTORIO_RAIZ / "Front"
CARPETA_MANEJO_DATOS = DIRECTORIO_RAIZ / "Manejo de los datos"

for ruta in [CARPETA_FRONT, CARPETA_MANEJO_DATOS]:
    if str(ruta) not in sys.path:
        sys.path.insert(0, str(ruta))


def main():
    # Comprobar si se solicitó ejecutar la etapa 1 de exploración de datos
    if "--etapa1" in sys.argv:
        from main_manejo_datos import ejecutar_etapa_exploracion
        ejecutar_etapa_exploracion()
        return

    print("=" * 75)
    print("PLATAFORMA PREDICTIVA DE CIENCIA DE MATERIALES (MPEAs)")
    print("Modelos XGBoost: Módulo de Young & Límite Elástico")
    print("=" * 75)
    print("\nIniciando servidor web y abriendo interfaz en el navegador...")

    from app import iniciar_servidor
    # Inicia el servidor Flask y abre el navegador por defecto
    iniciar_servidor(abrir_navegador=True, puerto=5000)


if __name__ == "__main__":
    main()