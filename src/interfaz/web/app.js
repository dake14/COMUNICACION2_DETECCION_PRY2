const $ = (id) => document.getElementById(id);
const COLOR = { fondo: "#1a1a19", rejilla: "#2d2d2a", eje: "#4a4a44", texto: "#98978c", azul: "57,135,229", naranja: "217,89,38" };
const ESTADO = { archivo: null, datos: null, vista: "recuperada", ultimo: null, ocupado: false };
const NOMBRES_RUTA = { red: "Red", clasica: "CMA", directa: "Filtro adaptado" };

function lienzo(canvas) {
  const caja = canvas.getBoundingClientRect();
  const escala = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(caja.width * escala));
  canvas.height = Math.max(1, Math.round(caja.height * escala));
  const g = canvas.getContext("2d");
  g.setTransform(escala, 0, 0, escala, 0, 0);
  g.fillStyle = COLOR.fondo;
  g.fillRect(0, 0, caja.width, caja.height);
  return { g, ancho: caja.width, alto: caja.height };
}

function escalas({ ancho, alto }, xMin, xMax, yMin, yMax, margen) {
  const area = { izq: margen.izq, der: ancho - margen.der, arr: margen.arr, aba: alto - margen.aba };
  return {
    area,
    x: (v) => area.izq + ((v - xMin) / (xMax - xMin)) * (area.der - area.izq),
    y: (v) => area.aba - ((v - yMin) / (yMax - yMin)) * (area.aba - area.arr),
  };
}

function linea(g, x1, y1, x2, y2, color) {
  g.strokeStyle = color;
  g.lineWidth = 1;
  g.beginPath();
  g.moveTo(Math.round(x1) + 0.5, Math.round(y1) + 0.5);
  g.lineTo(Math.round(x2) + 0.5, Math.round(y2) + 0.5);
  g.stroke();
}

function textoEje(g, texto, x, y, alineacion = "center") {
  g.fillStyle = COLOR.texto;
  g.font = "11px Segoe UI, system-ui, sans-serif";
  g.textAlign = alineacion;
  g.textBaseline = "middle";
  g.fillText(texto, x, y);
}

function dibujarConstelacion() {
  const canvas = $("c-constelacion");
  const contexto = lienzo(canvas);
  const { g } = contexto;
  const fuente = ESTADO.vista === "entrada" ? ESTADO.datos.entrada : ESTADO.datos.constelacion;
  const limite = 2.2;
  const e = escalas(contexto, -limite, limite, -limite, limite, { izq: 30, der: 10, arr: 10, aba: 24 });
  for (const v of [-2, -1, 0, 1, 2]) {
    const color = v === 0 ? COLOR.eje : COLOR.rejilla;
    linea(g, e.x(v), e.area.arr, e.x(v), e.area.aba, color);
    linea(g, e.area.izq, e.y(v), e.area.der, e.y(v), color);
    textoEje(g, String(v), e.x(v), e.area.aba + 12);
    textoEje(g, String(v), e.area.izq - 6, e.y(v), "right");
  }
  g.save();
  g.beginPath();
  g.rect(e.area.izq, e.area.arr, e.area.der - e.area.izq, e.area.aba - e.area.arr);
  g.clip();
  g.globalCompositeOperation = "lighter";
  g.fillStyle = `rgba(${COLOR.azul},0.42)`;
  const lado = contexto.ancho > 420 ? 2.4 : 2;
  (fuente?.x || []).forEach((x, i) => g.fillRect(e.x(x) - lado / 2, e.y(fuente.y[i]) - lado / 2, lado, lado));
  g.restore();
  $("p-constelacion").textContent = ESTADO.vista === "entrada"
    ? `${fuente.x.length} muestras de la señal sin recuperar.`
    : `${fuente?.x.length || 0} símbolos recuperados, con potencia unitaria.`;
}

function dibujarOjo(id, trazas, tiempo, rgb) {
  const contexto = lienzo($(id));
  const { g } = contexto;
  const limite = 2;
  const e = escalas(contexto, -1, 1, -limite, limite, { izq: 30, der: 10, arr: 8, aba: 20 });
  for (const v of [-2, -1, 0, 1, 2]) {
    linea(g, e.area.izq, e.y(v), e.area.der, e.y(v), v === 0 ? COLOR.eje : COLOR.rejilla);
    textoEje(g, String(v), e.area.izq - 6, e.y(v), "right");
  }
  for (const v of [-1, -0.5, 0, 0.5, 1]) {
    linea(g, e.x(v), e.area.arr, e.x(v), e.area.aba, v === 0 ? COLOR.eje : COLOR.rejilla);
    textoEje(g, v === 0 ? "0" : `${v}T`, e.x(v), e.area.aba + 11);
  }
  g.save();
  g.beginPath();
  g.rect(e.area.izq, e.area.arr, e.area.der - e.area.izq, e.area.aba - e.area.arr);
  g.clip();
  g.globalCompositeOperation = "lighter";
  g.strokeStyle = `rgba(${rgb},0.2)`;
  g.lineWidth = 1;
  for (const traza of trazas) {
    g.beginPath();
    traza.forEach((v, i) => (i ? g.lineTo(e.x(tiempo[i]), e.y(v)) : g.moveTo(e.x(tiempo[i]), e.y(v))));
    g.stroke();
  }
  g.restore();
}

function rampa(v) {
  const t = v / 255;
  return [26 + t * 174, 26 + t * 200, 25 + t * 226];
}

function dibujarImagen(id, matriz) {
  const canvas = $(id);
  const lado = matriz.length;
  canvas.width = lado;
  canvas.height = lado;
  const g = canvas.getContext("2d");
  const imagen = g.createImageData(lado, lado);
  matriz.forEach((fila, y) => fila.forEach((v, x) => {
    const i = ((lado - 1 - y) * lado + x) * 4;
    const [r, v2, a] = rampa(v);
    imagen.data.set([r, v2, a, 255], i);
  }));
  g.putImageData(imagen, 0, 0);
  canvas.dataset.lado = lado;
  canvas._matriz = matriz;
}

function colorHuella(t) {
  const fuerza = Math.min(1, Math.abs(t));
  const base = [58, 58, 54];
  const polo = t >= 0 ? [57, 135, 229] : [217, 89, 38];
  return `rgb(${base.map((b, i) => Math.round(b + (polo[i] - b) * fuerza)).join(",")})`;
}

function dibujarHuella(huella) {
  const contexto = lienzo($("c-huella"));
  const { g, ancho, alto } = contexto;
  const maximo = Math.max(...huella.map(Math.abs), 1e-6);
  const paso = ancho / huella.length;
  huella.forEach((v, i) => {
    g.fillStyle = colorHuella(v / maximo);
    g.fillRect(i * paso, 0, Math.max(1, paso - 1), alto);
  });
}

function textoPorcentaje(v) {
  return `${(v * 100).toFixed(1)} %`;
}

function pintarProbabilidades(d) {
  const lista = $("probabilidades");
  lista.replaceChildren();
  const maximo = Math.max(...d.probabilidades.map((p) => p.valor));
  const muchas = d.probabilidades.length > 8;
  for (const p of d.probabilidades) {
    if (muchas && p.valor < 0.005 && p.nombre !== "DESCONOCIDA") continue;
    const li = document.createElement("li");
    li.classList.toggle("elegida", p.valor === maximo);
    li.classList.toggle("desconocida", p.nombre === "DESCONOCIDA");
    li.dataset.globo = `${p.nombre}: ${textoPorcentaje(p.valor)}`;
    const etiqueta = document.createElement("span");
    etiqueta.className = "etiqueta";
    etiqueta.textContent = p.nombre === "DESCONOCIDA" ? "Desconocida" : p.nombre;
    const pista = document.createElement("span");
    pista.className = "pista";
    const relleno = document.createElement("span");
    relleno.className = "relleno";
    relleno.style.display = "block";
    relleno.style.width = `${p.valor * 100}%`;
    pista.append(relleno);
    const valor = document.createElement("span");
    valor.className = "valor";
    valor.textContent = textoPorcentaje(p.valor);
    li.append(etiqueta, pista, valor);
    lista.append(li);
  }
  $("p-umbral").textContent = `Responde una modulación si su probabilidad llega a ${textoPorcentaje(d.decision.umbral)}.`;
}

const NOMBRES_FAMILIA = { amplitud: "Amplitud", fase: "Fase", qam: "QAM", apsk: "APSK", frecuencia: "Frecuencia" };
const NOMBRES_VISTA = { espectro: "Espectro", potencias: "Potencias", envolvente: "Envolvente", frecuencia: "Frecuencia",
                        constelacion: "Constelación (QAM/APSK)" };

// CONTRATO (unidad U11, cerrado). Rellenar solo el cuerpo de pintarEnrutador.
// d es la respuesta de /api/analizar o /api/ejemplo. Con un motor sin enrutador, d.enrutador no existe:
//   ocultar document.querySelector(".panel-familias") (hidden = true) y salir.
// Si existe, mostrar el panel (hidden = false). Forma:
//   d.enrutador = { familias: [{ nombre, valor }] (5, ordenadas de mayor a menor valor),
//                   especialistas: [{ vista, clase, valor }] (4, en el orden espectro, potencias, envolvente, frecuencia),
//                   rama: "constelacion" | "frecuencia",
//                   tonos: null | { tonos: int, h: float, sps: float | null } }
// - #familias: vaciar y crear un <li> por familia con la misma estructura que pintarProbabilidades:
//   span.etiqueta (NOMBRES_FAMILIA[nombre] || nombre), span.pista > span.relleno (display block, width valor*100 %),
//   span.valor (textoPorcentaje(valor)); clase "elegida" en la de mayor valor;
//   dataset.globo = `${NOMBRES_FAMILIA[nombre] || nombre}: ${textoPorcentaje(valor)}`.
// - #p-rama: con rama "constelacion": "La constelación decide la modulación dentro de la familia.";
//   con "frecuencia": "Familia de frecuencia: deciden los especialistas y el conteo de tonos."
// - #especialistas: vaciar y crear un <li> por especialista con span.vista (NOMBRES_VISTA[vista] || vista),
//   span.clase (clase) y span.valor (textoPorcentaje(valor)).
// - #p-tonos: si tonos es null, hidden = true. Si no, hidden = false y el texto
//   `${tonos.tonos} tonos · índice h ≈ ${tonos.h.toFixed(2)}`, más ` · ${tonos.sps.toFixed(1)} muestras por símbolo`
//   cuando sps no es null.
// - Solo createElement, textContent, classList, dataset, style y replaceChildren: nada de innerHTML. Sin dependencias.
// - No tocar ninguna otra función ni archivo.
function pintarEnrutador(d) {
  const panel = document.querySelector(".panel-familias");
  const enr = d?.enrutador;
  if (!enr) {
    panel.hidden = true;
    return;
  }
  panel.hidden = false;

  const familias = enr.familias || [];
  const maximo = familias.length ? Math.max(...familias.map((f) => f.valor)) : 0;
  $("familias").replaceChildren(...familias.map((f) => {
    const nombre = NOMBRES_FAMILIA[f.nombre] || f.nombre;
    const li = document.createElement("li");
    li.classList.toggle("elegida", f.valor === maximo);
    li.dataset.globo = `${nombre}: ${textoPorcentaje(f.valor)}`;
    const etiqueta = document.createElement("span");
    etiqueta.classList.add("etiqueta");
    etiqueta.textContent = nombre;
    const pista = document.createElement("span");
    pista.classList.add("pista");
    const relleno = document.createElement("span");
    relleno.classList.add("relleno");
    relleno.style.display = "block";
    relleno.style.width = `${f.valor * 100}%`;
    pista.replaceChildren(relleno);
    const valor = document.createElement("span");
    valor.classList.add("valor");
    valor.textContent = textoPorcentaje(f.valor);
    li.replaceChildren(etiqueta, pista, valor);
    return li;
  }));

  $("p-rama").textContent = enr.rama === "frecuencia"
    ? "Familia de frecuencia: deciden los especialistas y el conteo de tonos."
    : enr.rama === "constelacion"
      ? "La constelación decide la modulación dentro de la familia."
      : "";

  $("especialistas").replaceChildren(...(enr.especialistas || []).map((e) => {
    const li = document.createElement("li");
    const vista = document.createElement("span");
    vista.classList.add("vista");
    vista.textContent = NOMBRES_VISTA[e.vista] || e.vista;
    const clase = document.createElement("span");
    clase.classList.add("clase");
    clase.textContent = e.clase;
    const valor = document.createElement("span");
    valor.classList.add("valor");
    valor.textContent = textoPorcentaje(e.valor);
    li.replaceChildren(vista, clase, valor);
    return li;
  }));

  const pTonos = $("p-tonos");
  if (!enr.tonos) {
    pTonos.hidden = true;
    pTonos.textContent = "";
  } else {
    pTonos.hidden = false;
    const fiabilidad = enr.tonos.fiable === false ? " (conteo poco fiable: no se usa para decidir)" : "";
    const base = `${enr.tonos.tonos} tonos${fiabilidad} · índice h ≈ ${enr.tonos.h.toFixed(2)}`;
    pTonos.textContent = enr.tonos.sps != null
      ? `${base} · ${enr.tonos.sps.toFixed(1)} muestras por símbolo`
      : base;
  }
}

// CONTRATO (unidad U12, cerrado). Rellenar solo el cuerpo de pintarAlternativas.
// d.decision puede traer (solo el motor integrado): mas_probable (string), misma_senal_que (lista de strings) y
// tambien_puede_ser (lista de { nombre, probabilidad }, ya ordenada de mayor a menor). Cualquiera puede faltar.
// - #d-alternativas: se arma con dos partes, en este orden, y queda hidden si no hay ninguna:
//   1. Si d.decision.modulacion es "INDETERMINADA" y hay mas_probable: un <strong> con
//      `Más probable: ${nombreModulacion(mas_probable)}` seguido del texto ` (${textoPorcentaje(d.decision.confianza)})`.
//   2. Si tambien_puede_ser no está vacía: el texto "También puede ser: " (con ". " delante si hubo parte 1) y luego,
//      separados por ", ", `${nombreModulacion(a.nombre)} (${textoPorcentaje(a.probabilidad)})` por cada alternativa.
// - #d-equivalentes: si misma_senal_que no está vacía, el texto
//   `Es la misma señal que ${lista unida por ", "}: no se pueden distinguir.` y hidden = false; si no, vaciar y
//   hidden = true.
// - Vaciar ambos elementos antes de escribir (replaceChildren). Solo createElement, textContent, append y
//   replaceChildren: nada de innerHTML. Sin dependencias. No tocar ninguna otra función ni archivo.
function pintarAlternativas(d) {
  const dec = d?.decision || {};
  const alt = $("d-alternativas");
  const eq = $("d-equivalentes");
  alt.replaceChildren();
  eq.replaceChildren();

  const parte1 = dec.modulacion === "INDETERMINADA" && Boolean(dec.mas_probable);
  const alternativas = Array.isArray(dec.tambien_puede_ser) ? dec.tambien_puede_ser : [];

  if (parte1) {
    const fuerte = document.createElement("strong");
    fuerte.textContent = `Más probable: ${nombreModulacion(dec.mas_probable)}`;
    alt.append(fuerte, ` (${textoPorcentaje(dec.confianza)})`);
  }
  if (alternativas.length > 0) {
    const lista = alternativas
      .map((a) => `${nombreModulacion(a.nombre)} (${textoPorcentaje(a.probabilidad)})`)
      .join(", ");
    alt.append(`${parte1 ? ". " : ""}También puede ser: ${lista}`);
  }
  alt.hidden = !(parte1 || alternativas.length > 0);

  const equivalentes = Array.isArray(dec.misma_senal_que) ? dec.misma_senal_que : [];
  if (equivalentes.length > 0) {
    eq.textContent = `Es la misma señal que ${equivalentes.join(", ")}: no se pueden distinguir.`;
    eq.hidden = false;
  } else {
    eq.hidden = true;
  }
  pintarNoCabe(dec);
}

function pintarNoCabe(dec) {
  const nota = $("d-no-cabe");
  const clases = Array.isArray(dec.no_cabe_en) ? dec.no_cabe_en : [];
  nota.textContent = clases.length
    ? `Los símbolos no caben en la constelación ideal de ${clases.join(", ")}: por eso sube «Desconocida».`
    : "";
  nota.hidden = clases.length === 0;
}

function nombreModulacion(m) {
  return { DESCONOCIDA: "Desconocida", INDETERMINADA: "Sin decidir" }[m] || m;
}

function marcaVerdad(d) {
  if (!d.verdad) return null;
  const real = d.verdad.etiqueta;
  const respuesta = d.decision.modulacion;
  const marca = document.createElement("span");
  marca.className = "marca";
  if (respuesta === "INDETERMINADA") {
    marca.classList.add("aviso");
    marca.textContent = `Sin decidir. Real: ${d.verdad.modulacion}`;
  } else if (respuesta === real) {
    marca.classList.add("bien");
    marca.textContent = `✓ Correcto. Real: ${d.verdad.modulacion}`;
  } else {
    marca.classList.add("mal");
    marca.textContent = `✗ Incorrecto. Real: ${d.verdad.modulacion}`;
  }
  return marca;
}

function pintarDecision(d) {
  $("d-modulacion").textContent = nombreModulacion(d.decision.modulacion);
  const detalle = $("d-detalle");
  detalle.replaceChildren();
  const partes = [];
  if (d.decision.confianza !== null) partes.push(`Confianza ${textoPorcentaje(d.decision.confianza)}`);
  if (d.etapas?.detectada === false) partes.push("El detector no ve señal clara");
  if (d.aviso) partes.push(d.aviso);
  detalle.append(partes.join(" · "));
  const marca = marcaVerdad(d);
  if (marca) detalle.append(partes.length ? " · " : "", marca);
  const e = d.etapas;
  const filas = e ? [
    ["SNR estimada", e.snr_db === null ? "n/d" : `${e.snr_db} dB`],
    ["Muestras/símbolo", e.sps],
    ["Desvío de portadora", `${e.cfo.toExponential(1)} c/m`],
    ["Recuperación", NOMBRES_RUTA[e.ruta] || e.ruta],
    ["Símbolos", e.simbolos.toLocaleString("es")],
    ["Muestras usadas", d.muestras_analizadas.toLocaleString("es")],
    ["Nitidez", e.nitidez],
    ["Tiempo", `${e.tiempo_ms} ms`],
  ] : [];
  $("etapas").replaceChildren(...filas.flatMap(([a, b]) => {
    const envoltura = document.createElement("div");
    const dt = document.createElement("dt");
    dt.textContent = a;
    const dd = document.createElement("dd");
    dd.textContent = b;
    envoltura.append(dt, dd);
    return [envoltura];
  }));
}

function dibujarTodo() {
  const d = ESTADO.datos;
  if (!d) return;
  dibujarConstelacion();
  const ojo = d.ojo || { t: [], i: [], q: [] };
  dibujarOjo("c-ojo-i", ojo.i, ojo.t, COLOR.azul);
  dibujarOjo("c-ojo-q", ojo.q, ojo.t, COLOR.naranja);
  if (d.cnn) dibujarHuella(d.cnn.huella);
}

function mostrar(d) {
  ESTADO.datos = d;
  $("vacio").hidden = true;
  $("resultado").hidden = false;
  $("nombre-archivo").textContent = d.nombre;
  pintarDecision(d);
  pintarAlternativas(d);
  pintarEnrutador(d);
  $("v-recuperada").disabled = !d.constelacion;
  if (!d.constelacion) cambiarVista("entrada", false);
  document.querySelector(".panel-cnn").hidden = !d.cnn;
  document.querySelector(".panel-ojo").hidden = !d.ojo;
  if (d.cnn) {
    pintarProbabilidades(d);
    dibujarImagen("c-cart", d.cnn.entrada.cartesiana);
    dibujarImagen("c-polar", d.cnn.entrada.polar);
    $("p-geometria").textContent = `La CNN estima unos ${d.cnn.puntos_aprox} puntos y ${d.cnn.anillos_aprox} anillos de amplitud.`;
  }
  dibujarTodo();
}

function ponerOcupado(ocupado, mensaje) {
  ESTADO.ocupado = ocupado;
  $("estado-global").textContent = mensaje;
  actualizarBotones();
}

function actualizarBotones() {
  $("analizar").disabled = ESTADO.ocupado || !ESTADO.ultimo;
  $("b-cargar").disabled = ESTADO.ocupado;
  $("b-azar").disabled = ESTADO.ocupado;
}

function ponerError(mensaje) {
  $("error").hidden = !mensaje;
  $("error").textContent = mensaje || "";
}

function parametros(extra = {}) {
  const p = new URLSearchParams({ motor: $("motor").value, inicio: $("inicio").value.trim() || "0", ...extra });
  const sps = $("sps").value.trim();
  if (sps) p.set("sps", sps);
  return p;
}

async function pedir(url, opciones) {
  ponerError("");
  ponerOcupado(true, "Analizando…");
  try {
    const r = await fetch(url, opciones);
    const datos = await r.json();
    if (!r.ok) throw new Error(datos.error || "No se pudo analizar la señal.");
    mostrar(datos);
    ponerOcupado(false, `Listo: ${datos.nombre}`);
  } catch (error) {
    ponerError(error.message === "Failed to fetch" ? "No hay conexión con el servidor local." : error.message);
    ponerOcupado(false, "Revisa el aviso");
  }
}

function analizarArchivo() {
  const archivo = ESTADO.archivo;
  const p = parametros({ nombre: archivo.name, formato: $("formato").value });
  return pedir(`/api/analizar?${p}`, { method: "POST", body: archivo });
}

function analizarUltimo() {
  return ESTADO.ultimo?.();
}

function elegirArchivo(archivo) {
  if (!archivo) return;
  ESTADO.archivo = archivo;
  ESTADO.ultimo = analizarArchivo;
  $("nombre-archivo").textContent = archivo.name;
  analizarArchivo();
}

function cargarBanco(azar) {
  const seleccion = azar ? {} : {
    generador: $("b-generador").value, canal: $("b-canal").value,
    modulacion: $("b-modulacion").value, indice: $("b-indice").value,
  };
  ESTADO.ultimo = () => pedir(`/api/ejemplo?${parametros(seleccion)}`);
  return ESTADO.ultimo();
}

function llenar(select, opciones, valor) {
  select.replaceChildren(...opciones.map((o) => new Option(o.nombre, o.id, false, o.id === valor)));
}

function cambiarVista(vista, redibujar = true) {
  ESTADO.vista = vista;
  $("v-recuperada").setAttribute("aria-pressed", String(vista === "recuperada"));
  $("v-entrada").setAttribute("aria-pressed", String(vista === "entrada"));
  if (redibujar && ESTADO.datos) dibujarConstelacion();
}

function mostrarGlobo(evento, texto) {
  const globo = $("globo");
  globo.hidden = !texto;
  if (!texto) return;
  globo.textContent = texto;
  globo.style.left = `${Math.min(evento.clientX + 12, window.innerWidth - 150)}px`;
  globo.style.top = `${evento.clientY + 12}px`;
}

function globoImagen(evento, canvas, ejes) {
  const matriz = canvas._matriz;
  if (!matriz) return;
  const caja = canvas.getBoundingClientRect();
  const x = Math.min(63, Math.floor(((evento.clientX - caja.left) / caja.width) * 64));
  const y = Math.min(63, Math.floor(((caja.bottom - evento.clientY) / caja.height) * 64));
  mostrarGlobo(evento, `${ejes(x, y)} · densidad relativa ${(matriz[y][x] / 255).toFixed(2)}`);
}

function conectarGlobos() {
  $("probabilidades").addEventListener("mousemove", (e) => mostrarGlobo(e, e.target.closest("li")?.dataset.globo));
  $("probabilidades").addEventListener("mouseleave", (e) => mostrarGlobo(e, ""));
  $("familias").addEventListener("mousemove", (e) => mostrarGlobo(e, e.target.closest("li")?.dataset.globo));
  $("familias").addEventListener("mouseleave", (e) => mostrarGlobo(e, ""));
  const pos = (v) => ((v + 0.5) / 64) * 4.4 - 2.2;
  $("c-cart").addEventListener("mousemove", (e) => globoImagen(e, e.currentTarget, (x, y) => `I ${pos(x).toFixed(2)}, Q ${pos(y).toFixed(2)}`));
  $("c-polar").addEventListener("mousemove", (e) => globoImagen(e, e.currentTarget, (x, y) => `ángulo ${(((x + 0.5) / 64) * 360 - 180).toFixed(0)}°, radio ${(((y + 0.5) / 64) * 2.2).toFixed(2)}`));
  $("c-huella").addEventListener("mousemove", (e) => {
    const h = ESTADO.datos?.cnn?.huella;
    if (!h) return;
    const caja = e.currentTarget.getBoundingClientRect();
    const i = Math.min(h.length - 1, Math.floor(((e.clientX - caja.left) / caja.width) * h.length));
    mostrarGlobo(e, `Valor ${i + 1}: ${h[i].toFixed(3)}`);
  });
  for (const id of ["c-cart", "c-polar", "c-huella"]) $(id).addEventListener("mouseleave", (e) => mostrarGlobo(e, ""));
}

function conectarArrastre() {
  let profundidad = 0;
  const tieneArchivos = (e) => [...(e.dataTransfer?.types || [])].includes("Files");
  window.addEventListener("dragenter", (e) => {
    if (!tieneArchivos(e)) return;
    e.preventDefault();
    profundidad += 1;
    $("soltar").hidden = false;
  });
  window.addEventListener("dragover", (e) => { if (tieneArchivos(e)) e.preventDefault(); });
  window.addEventListener("dragleave", () => {
    profundidad = Math.max(0, profundidad - 1);
    if (!profundidad) $("soltar").hidden = true;
  });
  window.addEventListener("drop", (e) => {
    e.preventDefault();
    profundidad = 0;
    $("soltar").hidden = true;
    elegirArchivo(e.dataTransfer?.files?.[0]);
  });
}

function conectar() {
  $("archivo").addEventListener("change", (e) => elegirArchivo(e.target.files[0]));
  $("zona").addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("archivo").click(); }
  });
  $("analizar").addEventListener("click", analizarUltimo);
  $("b-cargar").addEventListener("click", () => cargarBanco(false));
  $("b-azar").addEventListener("click", () => cargarBanco(true));
  $("v-recuperada").addEventListener("click", () => cambiarVista("recuperada"));
  $("v-entrada").addEventListener("click", () => cambiarVista("entrada"));
  $("motor").addEventListener("change", ayudaMotor);
  conectarArrastre();
  conectarGlobos();
  new ResizeObserver(() => dibujarTodo()).observe($("resultado"));
}

let MOTORES = [];
function ayudaMotor() {
  $("motor-ayuda").textContent = MOTORES.find((m) => m.id === $("motor").value)?.descripcion || "";
}

async function iniciar() {
  conectar();
  actualizarBotones();
  try {
    const e = await (await fetch("/api/estado")).json();
    MOTORES = e.motores;
    llenar($("formato"), e.formatos, "auto");
    llenar($("motor"), e.motores.filter((m) => m.disponible), e.motor_defecto);
    ayudaMotor();
    if (e.banco) {
      const lista = (v) => v.map((x) => ({ id: String(x), nombre: String(x) }));
      llenar($("b-generador"), lista(e.banco.generadores), "procedural");
      llenar($("b-canal"), lista(e.banco.canales), "ideal");
      llenar($("b-modulacion"), lista(e.banco.modulaciones), "QPSK");
      llenar($("b-indice"), e.banco.indices.map((x) => ({ id: String(x), nombre: String(x + 1) })), "0");
      $("tarjeta-banco").hidden = false;
    }
    $("estado-global").textContent = "Listo para analizar";
  } catch {
    ponerError("No se pudo conectar con el servidor local.");
    $("estado-global").textContent = "Sin conexión";
  }
}

iniciar();
