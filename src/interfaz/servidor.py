# ─────────────────────────────  Imports  ─────────────────────────────
import argparse
import json
import threading
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from src.interfaz.analisis import LIMITE_MUESTRAS, analizar_senal
from src.interfaz.lectura import FORMATOS, leer_senal
from src.interfaz.recursos import MOTOR_DEFECTO, Banco, Modelos

# ─────────────────────────────  Config  ──────────────────────────────
ANFITRION = "127.0.0.1"
PUERTO = 8765
MAX_BYTES = 512 * 1024 * 1024
WEB = Path(__file__).with_name("web")
ESTATICOS = {"/": ("index.html", "text/html; charset=utf-8"),
             "/estilo.css": ("estilo.css", "text/css; charset=utf-8"),
             "/app.js": ("app.js", "text/javascript; charset=utf-8")}


# ───────────────────────────  Parámetros  ────────────────────────────
def _uno(consulta, clave, defecto=None):
    valor = consulta.get(clave, [""])[0].strip()
    return valor if valor else defecto


def _opciones_analisis(consulta):
    try:
        sps = _uno(consulta, "sps")
        sps = None if sps is None else float(sps.replace(",", "."))
        inicio = int(_uno(consulta, "inicio", "0"))
    except ValueError as error:
        raise ValueError("Las muestras por símbolo y la muestra inicial deben ser números.") from error
    if sps is not None and not 2 <= sps <= 64:
        raise ValueError("Las muestras por símbolo deben estar entre 2 y 64.")
    return _uno(consulta, "motor", MOTOR_DEFECTO), sps, inicio


# ─────────────────────────────  Handler  ─────────────────────────────
class Manejador(BaseHTTPRequestHandler):
    server_version = "AnalizadorModulaciones"

    def log_message(self, *argumentos):
        pass

    def _origen_valido(self):
        puerto = self.server.server_address[1]
        permitidos = {f"127.0.0.1:{puerto}", f"localhost:{puerto}"}
        if self.headers.get("Host") not in permitidos:
            return False
        origen = self.headers.get("Origin")
        return origen is None or urlsplit(origen).netloc in permitidos

    def _responder(self, codigo, cuerpo, tipo):
        self.send_response(codigo)
        self.send_header("Content-Type", tipo)
        self.send_header("Content-Length", str(len(cuerpo)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(cuerpo)

    def _json(self, codigo, datos):
        self._responder(codigo, json.dumps(datos, ensure_ascii=False, allow_nan=False).encode("utf-8"),
                        "application/json; charset=utf-8")

    def _proteger(self, accion):
        if not self._origen_valido():
            return self._json(403, {"error": "Origen no permitido."})
        try:
            return accion()
        except ValueError as error:
            return self._json(400, {"error": str(error)})
        except Exception:
            traceback.print_exc()
            return self._json(500, {"error": "Ocurrió un error interno al analizar la señal."})

    def _analizar(self, iq, consulta, extra):
        motor, sps, inicio = _opciones_analisis(consulta)
        paquete = self.server.modelos.obtener(motor)
        with self.server.cerrojo:
            resultado = analizar_senal(iq, paquete, sps, inicio)
        self._json(200, {**resultado, **extra, "motor": motor})

    def _estado(self):
        self._json(200, {"motores": self.server.modelos.listado(), "motor_defecto": MOTOR_DEFECTO,
                         "formatos": [{"id": k, "nombre": v} for k, v in FORMATOS.items()],
                         "banco": self.server.banco.opciones(), "limite_muestras": LIMITE_MUESTRAS})

    def _ejemplo(self, consulta):
        indice = _uno(consulta, "indice")
        iq, verdad = self.server.banco.ejemplo(_uno(consulta, "generador"), _uno(consulta, "canal"),
                                               _uno(consulta, "modulacion"), indice)
        nombre = f"{verdad['generador']} · {verdad['canal']} · {verdad['modulacion']} · {verdad['indice']}"
        self._analizar(iq, consulta, {"nombre": nombre, "verdad": verdad})

    def _subida(self, consulta):
        largo = int(self.headers.get("Content-Length") or 0)
        if largo > MAX_BYTES:
            self.close_connection = True
            raise ValueError("El archivo supera el límite de 512 MB.")
        datos = self.rfile.read(largo)
        nombre = _uno(consulta, "nombre", "senal")
        iq = leer_senal(datos, nombre, _uno(consulta, "formato", "auto"))
        self._analizar(iq, consulta, {"nombre": nombre})

    def do_GET(self):
        partes = urlsplit(self.path)
        consulta = parse_qs(partes.query)
        if partes.path in ESTATICOS:
            archivo, tipo = ESTATICOS[partes.path]
            return self._proteger(lambda: self._responder(200, (WEB / archivo).read_bytes(), tipo))
        if partes.path == "/api/estado":
            return self._proteger(self._estado)
        if partes.path == "/api/ejemplo":
            return self._proteger(lambda: self._ejemplo(consulta))
        return self._json(404, {"error": "No encontrado."})

    def do_POST(self):
        partes = urlsplit(self.path)
        if partes.path != "/api/analizar":
            return self._json(404, {"error": "No encontrado."})
        return self._proteger(lambda: self._subida(parse_qs(partes.query)))


# ─────────────────────────────  Servidor  ────────────────────────────
class Servidor(ThreadingHTTPServer):
    allow_reuse_address = False


def crear_servidor(puerto=PUERTO, modelos=None, banco=None):
    servidor = Servidor((ANFITRION, puerto), Manejador)
    servidor.modelos = modelos or Modelos()
    servidor.banco = banco or Banco()
    servidor.cerrojo = threading.Lock()
    return servidor


# ──────────────────────────────  Entry  ──────────────────────────────
def main():
    p = argparse.ArgumentParser(description="Interfaz para analizar señales I/Q.")
    p.add_argument("--puerto", type=int, default=PUERTO)
    p.add_argument("--sin-navegador", action="store_true")
    a = p.parse_args()
    direccion = f"http://{ANFITRION}:{a.puerto}/"
    try:
        servidor = crear_servidor(a.puerto)
    except OSError:
        print("La interfaz ya está abierta en", direccion, flush=True)
        if not a.sin_navegador:
            webbrowser.open(direccion)
        return
    print("Interfaz lista en", direccion, flush=True)
    if not a.sin_navegador:
        webbrowser.open(direccion)
    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        servidor.server_close()
