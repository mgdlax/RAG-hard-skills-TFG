"""
update_profiles.py — Re-indexa usuarios solo si han tenido actividad en GitHub.

Reutiliza services.profile_service.update_profiles(), la misma función que usa
el botón "Buscar cambios y actualizar" del frontend. Aquí solo se imprime el
progreso por consola.

Uso:
    python update_profiles.py
"""

from services.profile_service import update_profiles


def main():
    print("=" * 60)
    print("ACTUALIZACIÓN DE PERFILES")
    print("=" * 60)

    for mensaje, is_final, resultado in update_profiles():
        print(mensaje)
        if is_final and resultado is not None:
            print("=" * 60)
            print(f"RESUMEN: {len(resultado['updated'])} actualizados, "
                  f"{len(resultado['skipped'])} sin cambios")
            print("=" * 60)


if __name__ == "__main__":
    main()
