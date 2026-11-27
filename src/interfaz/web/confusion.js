"use strict";
// CONTRATO (unidad U16, cerrado). Página /confusion (confusion.html). Rellenar solo los cuerpos marcados con
// `throw new Error("sin implementar")`. Se pueden agregar funciones auxiliares en este archivo. Sin dependencias, sin
// innerHTML (solo createElement, textContent, classList, dataset, style, title, replaceChildren, append). No tocar
// ningún otro archivo.
//
// DATOS: GET /api/confusion devuelve
//   { particion: str, archivos: int, generado: "AAAA-MM-DD", umbral_integrado: number,
//     filas: [str],                       // modulaciones reales, en orden (18 conocidas y luego las desconocidas)
//     familia_de: {str: str},             // fila -> familia ("amplitud", "fase", "qam", "apsk", "frecuencia" o
//                                         //   "desconocida"); incluye todas las clases que aparecen en columnas
//     configuraciones: [{ id, nombre, descripcion,
//                         columnas: [str], // clases de la configuración + "DESCONOCIDA" + "INDETERMINADA" al final
//                         matrices: {"todas": [[int]], "snr>=10": [[int]], "snr<10": [[int]]},  // filas × columnas
//                         sin_umbral: null | { columnas: [str], matrices: {...igual...} } }] }
//   Si la respuesta no es ok, el cuerpo es { error: str }.
//
// FUNCIONES PURAS (exportadas en window.Confusion para probarlas):
// - vista(datos, config, snr, conUmbral, nivel) -> { filas: [str], columnas: [str], matriz: [[int]] }
//   - Fuente: si conUmbral es false y config.sin_umbral no es null, config.sin_umbral; si no, config.
//     matriz = fuente.matrices[snr], columnas = fuente.columnas, filas = datos.filas.
//   - nivel "clase": tal cual. nivel "familia": se agrupan filas y columnas con fam(x) = datos.familia_de[x] si existe,
//     si no x (así "DESCONOCIDA" e "INDETERMINADA" quedan como están). Orden de filas: el de la primera aparición de
//     cada familia recorriendo datos.filas. Orden de columnas: el de la primera aparición recorriendo las columnas,
//     salvo que "DESCONOCIDA" e "INDETERMINADA" van siempre al final, en ese orden. Cada celda es la suma.
//   - Al final se quitan las filas cuya suma es 0 (en cualquier nivel).
// - esCorrecta(datos, nivel, fila, columna) -> bool: true si columna === fila, o si la fila es desconocida y
//   columna === "DESCONOCIDA". Fila desconocida: en nivel "clase", datos.familia_de[fila] === "desconocida"; en nivel
//   "familia", fila === "desconocida".
// - metricas(datos, config, snr, conUmbral) -> { conocidas, desconocidas, sinDecidir, n }
//   Siempre a nivel "clase" (con vista(..., "clase")). Conocidas: filas con familia_de distinta de "desconocida".
//   conocidas = aciertos (esCorrecta) / total de filas conocidas; desconocidas = celdas "DESCONOCIDA" de filas
//   desconocidas / total de filas desconocidas; sinDecidir = celdas "INDETERMINADA" de filas conocidas / total de
//   filas conocidas; n = total de la matriz. Un cociente con denominador 0 da null.
//
// DOM (todo existe en confusion.html):
// - Al cargar: fetch("/api/confusion"). Error de red o respuesta no ok: #c-error con hidden = false y el texto del
//   error (cuerpo.error o el mensaje), y nada más.
// - #c-particion: `${particion} · ${archivos} archivos · ${generado}`.
// - #c-config: un <option> por configuración (value = id, texto = nombre), en el orden de datos.configuraciones.
// - Estado: { config, snr, nivel, valores, conUmbral } leído de los controles (#c-config, #c-snr, #c-nivel,
//   #c-valores, #c-umbral.checked). Cualquier "change" redibuja todo. Guardar el estado en localStorage con la clave
//   "confusion" (JSON) y restaurarlo al cargar si los valores son válidos; todo acceso a localStorage va en try/catch.
// - #c-umbral: disabled = true cuando la configuración elegida tiene sin_umbral null (el estado conserva el valor).
// - #c-descripcion: descripcion de la configuración.
// - #c-titulo: el nombre de la configuración, más " · sin umbral" si se usa sin_umbral.
// - #c-resumen: con metricas(...): `Conocidas correctas ${p(conocidas)} · Desconocidas detectadas
//   ${p(desconocidas)} · Sin decidir ${p(sinDecidir)} · ${n} archivos`, donde p(x) = x === null ? "—" :
//   `${(100 * x).toFixed(1)} %`.
// - #c-tabla (vaciar y rearmar con vista(...)):
//   - <caption> con el mismo texto que #c-titulo.
//   - <thead>: una fila con un <th> vacío ("Real \\ Respuesta" como title), un <th class="col"> por columna cuyo
//     único hijo es un <span> con nombre(columna), y un <th class="total"> con texto "n".
//   - <tbody>: una <tr> por fila: <th scope="row"> con nombre(fila); un <td> por columna; <td class="total"> con la
//     suma de la fila.
//   - Celda con cuenta c y total de fila t (t > 0): f = c / t. Texto: si c === 0, ""; si no, con valores "fila"
//     Math.round(100 * f) (sin "%"; "<1" si redondea a 0), con "cuenta" String(c). Fondo: c === 0 -> sin estilo; si no
//     style.background = `rgba(57, 135, 229, ${(0.12 + 0.88 * f).toFixed(3)})` y, si f >= 0.5, style.color =
//     "#ffffff". classList "correcta" si esCorrecta(...). title = `Real: ${nombre(fila)} → Respuesta:
//     ${nombre(columna)} · ${c} de ${t} (${(100 * f).toFixed(1)} %)`.
//   - nombre(x): {DESCONOCIDA: "Desconocida", INDETERMINADA: "Sin decidir", amplitud: "Amplitud", fase: "Fase",
//     qam: "QAM", apsk: "APSK", frecuencia: "Frecuencia", desconocida: "Desconocidas"}[x] || x.
// - #c-comparacion (vaciar y rearmar): <thead> con <th> "Versión", "Conocidas", "Desconoc.", "Sin decidir"; <tbody>
//   con una <tr> por configuración (dataset.id = id) con su nombre y metricas(datos, config, estado.snr,
//   estado.conUmbral) formateadas con p(x). La de la configuración elegida lleva classList "actual". Clic en una fila:
//   elige esa configuración en #c-config y redibuja.
//
// CRITERIO DE ACEPTACIÓN (con un objeto de prueba hecho a mano en node o en la consola, filas ["BPSK", "QPSK",
// "12PSK"], familia_de {BPSK: "fase", QPSK: "fase", "12PSK": "desconocida"}, una configuración con columnas
// ["BPSK", "QPSK", "DESCONOCIDA", "INDETERMINADA"] y matriz "todas" [[8, 1, 0, 1], [2, 6, 1, 1], [0, 3, 5, 2]]):
// 1. metricas da conocidas 14/20 = 0.7, desconocidas 5/10 = 0.5, sinDecidir 2/20 = 0.1, n 30.
// 2. vista(..., "familia") da filas ["fase", "desconocida"], columnas ["fase", "DESCONOCIDA", "INDETERMINADA"] y
//    matriz [[17, 1, 2], [3, 5, 2]]; esCorrecta en familia: ("fase", "fase") y ("desconocida", "DESCONOCIDA") true.
// 3. Con conUmbral false y sin_umbral null se usa la matriz con umbral; con sin_umbral presente, la suya.
// 4. Una fila de suma 0 no aparece en la vista.
// 5. La página carga sin errores en la consola contra el servidor real y cambia de tabla al cambiar cada control.

(() => {
  const $ = (id) => document.getElementById(id);

  const CLAVE_ESTADO = "confusion";
  const NOMBRES = {
    DESCONOCIDA: "Desconocida",
    INDETERMINADA: "Sin decidir",
    amplitud: "Amplitud",
    fase: "Fase",
    qam: "QAM",
    apsk: "APSK",
    frecuencia: "Frecuencia",
    desconocida: "Desconocidas",
  };

  function nombre(x) {
    return Object.prototype.hasOwnProperty.call(NOMBRES, x) ? NOMBRES[x] : x;
  }

  function p(x) {
    return x === null ? "—" : `${(100 * x).toFixed(1)} %`;
  }

  function usaSinUmbral(config, conUmbral) {
    return conUmbral === false && config.sin_umbral !== null && config.sin_umbral !== undefined;
  }

  function sumaFila(fila) {
    let total = 0;
    for (let i = 0; i < fila.length; i++) {
      total += fila[i];
    }
    return total;
  }

  function filtrarFilasVacias(filas, matriz) {
    const filasOut = [];
    const matrizOut = [];
    for (let i = 0; i < filas.length; i++) {
      if (sumaFila(matriz[i]) !== 0) {
        filasOut.push(filas[i]);
        matrizOut.push(matriz[i]);
      }
    }
    return { filas: filasOut, matriz: matrizOut };
  }

  function vista(datos, config, snr, conUmbral, nivel) {
    const fuente = usaSinUmbral(config, conUmbral) ? config.sin_umbral : config;
    const matrizBase = fuente.matrices[snr];
    const columnasBase = fuente.columnas;
    const filasBase = datos.filas;

    if (nivel !== "familia") {
      const filasCopia = [];
      const matrizCopia = [];
      for (let i = 0; i < filasBase.length; i++) {
        const copia = matrizBase[i].slice();
        if (sumaFila(copia) !== 0) {
          filasCopia.push(filasBase[i]);
          matrizCopia.push(copia);
        }
      }
      return {
        filas: filasCopia,
        columnas: columnasBase.slice(),
        matriz: matrizCopia,
      };
    }

    const fam = (x) =>
      Object.prototype.hasOwnProperty.call(datos.familia_de, x)
        ? datos.familia_de[x]
        : x;

    const filasFam = [];
    const idxFila = new Map();
    for (let i = 0; i < filasBase.length; i++) {
      const f = fam(filasBase[i]);
      if (!idxFila.has(f)) {
        idxFila.set(f, filasFam.length);
        filasFam.push(f);
      }
    }

    const colsNormales = [];
    const vistasCol = new Set();
    let tieneDesconocida = false;
    let tieneIndeterminada = false;
    for (let j = 0; j < columnasBase.length; j++) {
      const c = fam(columnasBase[j]);
      if (c === "DESCONOCIDA") {
        tieneDesconocida = true;
      } else if (c === "INDETERMINADA") {
        tieneIndeterminada = true;
      } else if (!vistasCol.has(c)) {
        vistasCol.add(c);
        colsNormales.push(c);
      }
    }
    const colsFam = colsNormales.slice();
    if (tieneDesconocida) colsFam.push("DESCONOCIDA");
    if (tieneIndeterminada) colsFam.push("INDETERMINADA");

    const idxCol = new Map();
    for (let j = 0; j < colsFam.length; j++) {
      idxCol.set(colsFam[j], j);
    }

    const acum = Array.from({ length: filasFam.length }, () =>
      new Array(colsFam.length).fill(0)
    );
    for (let i = 0; i < filasBase.length; i++) {
      const fi = idxFila.get(fam(filasBase[i]));
      const origen = matrizBase[i];
      const destino = acum[fi];
      for (let j = 0; j < columnasBase.length; j++) {
        const cj = idxCol.get(fam(columnasBase[j]));
        destino[cj] += origen[j];
      }
    }

    const filtradas = filtrarFilasVacias(filasFam, acum);
    return {
      filas: filtradas.filas,
      columnas: colsFam,
      matriz: filtradas.matriz,
    };
  }

  function esCorrecta(datos, nivel, fila, columna) {
    if (columna === fila) return true;
    const filaDesconocida =
      nivel === "familia"
        ? fila === "desconocida"
        : datos.familia_de[fila] === "desconocida";
    return filaDesconocida && columna === "DESCONOCIDA";
  }

  function metricas(datos, config, snr, conUmbral) {
    const v = vista(datos, config, snr, conUmbral, "clase");
    const idxDesc = v.columnas.indexOf("DESCONOCIDA");
    const idxIndet = v.columnas.indexOf("INDETERMINADA");

    let aciertosConocidas = 0;
    let totalConocidas = 0;
    let aciertosDesconocidas = 0;
    let totalDesconocidas = 0;
    let indetConocidas = 0;
    let n = 0;

    for (let i = 0; i < v.filas.length; i++) {
      const fila = v.filas[i];
      const celdas = v.matriz[i];
      const esDesc = datos.familia_de[fila] === "desconocida";
      let totalFila = 0;

      for (let j = 0; j < v.columnas.length; j++) {
        const c = celdas[j];
        totalFila += c;
        if (!esDesc && esCorrecta(datos, "clase", fila, v.columnas[j])) {
          aciertosConocidas += c;
        }
      }

      n += totalFila;
      if (esDesc) {
        totalDesconocidas += totalFila;
        if (idxDesc !== -1) {
          aciertosDesconocidas += celdas[idxDesc];
        }
      } else {
        totalConocidas += totalFila;
        if (idxIndet !== -1) {
          indetConocidas += celdas[idxIndet];
        }
      }
    }

    return {
      conocidas: totalConocidas > 0 ? aciertosConocidas / totalConocidas : null,
      desconocidas:
        totalDesconocidas > 0 ? aciertosDesconocidas / totalDesconocidas : null,
      sinDecidir: totalConocidas > 0 ? indetConocidas / totalConocidas : null,
      n,
    };
  }

  function crearEl(tag, texto, clase) {
    const nodo = document.createElement(tag);
    if (texto !== undefined) nodo.textContent = texto;
    if (clase) nodo.classList.add(clase);
    return nodo;
  }

  function mostrarError(mensaje) {
    const alerta = $("c-error");
    alerta.textContent = mensaje;
    alerta.hidden = false;
  }

  function tieneOpcion(select, valor) {
    if (typeof valor !== "string") return false;
    for (let i = 0; i < select.options.length; i++) {
      if (select.options[i].value === valor) return true;
    }
    return false;
  }

  function leerEstado() {
    return {
      config: $("c-config").value,
      snr: $("c-snr").value,
      nivel: $("c-nivel").value,
      valores: $("c-valores").value,
      conUmbral: $("c-umbral").checked,
    };
  }

  function guardarEstado(estado) {
    try {
      localStorage.setItem(CLAVE_ESTADO, JSON.stringify(estado));
    } catch (_) {
      // Ignorar errores de almacenamiento.
    }
  }

  function restaurarEstado() {
    let guardado = null;
    try {
      const texto = localStorage.getItem(CLAVE_ESTADO);
      if (texto) guardado = JSON.parse(texto);
    } catch (_) {
      return;
    }
    if (!guardado || typeof guardado !== "object") return;

    const selConfig = $("c-config");
    const selSnr = $("c-snr");
    const selNivel = $("c-nivel");
    const selValores = $("c-valores");
    const chkUmbral = $("c-umbral");

    if (tieneOpcion(selConfig, guardado.config)) selConfig.value = guardado.config;
    if (tieneOpcion(selSnr, guardado.snr)) selSnr.value = guardado.snr;
    if (tieneOpcion(selNivel, guardado.nivel)) selNivel.value = guardado.nivel;
    if (tieneOpcion(selValores, guardado.valores)) selValores.value = guardado.valores;
    if (typeof guardado.conUmbral === "boolean") chkUmbral.checked = guardado.conUmbral;
  }

  function dibujarTabla(datos, config, estado, titulo) {
    const v = vista(datos, config, estado.snr, estado.conUmbral, estado.nivel);
    const tabla = $("c-tabla");

    const caption = crearEl("caption", titulo);

    const thead = document.createElement("thead");
    const trHead = document.createElement("tr");
    const thEsquina = document.createElement("th");
    thEsquina.title = "Real \\ Respuesta";
    trHead.append(thEsquina);

    for (let j = 0; j < v.columnas.length; j++) {
      const th = crearEl("th", undefined, "col");
      th.append(crearEl("span", nombre(v.columnas[j])));
      trHead.append(th);
    }
    trHead.append(crearEl("th", "n", "total"));
    thead.append(trHead);

    const tbody = document.createElement("tbody");
    for (let i = 0; i < v.filas.length; i++) {
      const fila = v.filas[i];
      const celdas = v.matriz[i];
      const t = sumaFila(celdas);
      const tr = document.createElement("tr");

      const thFila = crearEl("th", nombre(fila));
      thFila.scope = "row";
      tr.append(thFila);

      for (let j = 0; j < v.columnas.length; j++) {
        const columna = v.columnas[j];
        const c = celdas[j];
        const f = c / t;
        const td = document.createElement("td");

        if (c === 0) {
          td.textContent = "";
        } else {
          td.textContent =
            estado.valores === "fila"
              ? (Math.round(100 * f) === 0 ? "<1" : String(Math.round(100 * f)))
              : String(c);
          td.style.background = `rgba(57, 135, 229, ${(0.12 + 0.88 * f).toFixed(3)})`;
          if (f >= 0.5) {
            td.style.color = "#ffffff";
          }
        }

        if (esCorrecta(datos, estado.nivel, fila, columna)) {
          td.classList.add("correcta");
        }
        td.title = `Real: ${nombre(fila)} → Respuesta: ${nombre(columna)} · ${c} de ${t} (${(100 * f).toFixed(1)} %)`;
        tr.append(td);
      }

      tr.append(crearEl("td", String(t), "total"));
      tbody.append(tr);
    }

    tabla.replaceChildren(caption, thead, tbody);
  }

  function dibujarComparacion(datos, estado, alElegirConfig) {
    const tabla = $("c-comparacion");
    const thead = document.createElement("thead");
    const trHead = document.createElement("tr");
    const encabezados = ["Versión", "Conocidas", "Desconoc.", "Sin decidir"];
    for (let i = 0; i < encabezados.length; i++) {
      trHead.append(crearEl("th", encabezados[i]));
    }
    thead.append(trHead);

    const tbody = document.createElement("tbody");
    for (let i = 0; i < datos.configuraciones.length; i++) {
      const cfg = datos.configuraciones[i];
      const m = metricas(datos, cfg, estado.snr, estado.conUmbral);
      const tr = document.createElement("tr");
      tr.dataset.id = cfg.id;
      if (cfg.id === estado.config) {
        tr.classList.add("actual");
      }

      tr.append(
        crearEl("td", cfg.nombre),
        crearEl("td", p(m.conocidas)),
        crearEl("td", p(m.desconocidas)),
        crearEl("td", p(m.sinDecidir))
      );

      tr.addEventListener("click", () => {
        alElegirConfig(cfg.id);
      });
      tbody.append(tr);
    }

    tabla.replaceChildren(thead, tbody);
  }

  async function iniciar() {
    let datos;
    try {
      const resp = await fetch("/api/confusion");
      if (!resp.ok) {
        let mensaje = `Error HTTP ${resp.status}`;
        try {
          const cuerpo = await resp.json();
          if (cuerpo && typeof cuerpo.error === "string" && cuerpo.error) {
            mensaje = cuerpo.error;
          }
        } catch (_) {
          // Cuerpo no JSON.
        }
        mostrarError(mensaje);
        return;
      }
      datos = await resp.json();
    } catch (err) {
      mostrarError(err && err.message ? err.message : String(err));
      return;
    }

    $("c-particion").textContent =
      `${datos.particion} · ${datos.archivos} archivos · ${datos.generado}`;

    const selConfig = $("c-config");
    const opciones = datos.configuraciones.map((cfg) => {
      const opt = crearEl("option", cfg.nombre);
      opt.value = cfg.id;
      return opt;
    });
    selConfig.replaceChildren(...opciones);

    restaurarEstado();

    const porId = new Map(datos.configuraciones.map((cfg) => [cfg.id, cfg]));

    function redibujar() {
      const estado = leerEstado();
      const config = porId.get(estado.config) || datos.configuraciones[0];
      if (!config) return;
      estado.config = config.id;

      $("c-umbral").disabled = config.sin_umbral === null;
      $("c-descripcion").textContent = config.descripcion;

      const titulo = usaSinUmbral(config, estado.conUmbral)
        ? `${config.nombre} · sin umbral`
        : config.nombre;
      $("c-titulo").textContent = titulo;

      const m = metricas(datos, config, estado.snr, estado.conUmbral);
      $("c-resumen").textContent =
        `Conocidas correctas ${p(m.conocidas)} · Desconocidas detectadas ${p(m.desconocidas)} · Sin decidir ${p(m.sinDecidir)} · ${m.n} archivos`;

      dibujarTabla(datos, config, estado, titulo);
      dibujarComparacion(datos, estado, (id) => {
        selConfig.value = id;
        alCambiar();
      });
    }

    function alCambiar() {
      guardarEstado(leerEstado());
      redibujar();
    }

    const controles = ["c-config", "c-snr", "c-nivel", "c-valores", "c-umbral"];
    for (let i = 0; i < controles.length; i++) {
      $(controles[i]).addEventListener("change", alCambiar);
    }

    redibujar();
  }

  window.Confusion = { vista, esCorrecta, metricas };
  if (typeof document !== "undefined" && document.getElementById("c-tabla")) {
    iniciar();
  }
})();
