const repo = 'https://github.com/al073016/Practica-4-Modelos-de-machine-learning';
const defs = {
  bank: {
    title:'¿Contratará un depósito?', kicker:'CLASIFICACIÓN · BANCA',
    intro:'Una red neuronal analiza datos de campañas bancarias y estima la probabilidad de que un cliente contrate un depósito. La duración de la llamada no entra en el modelo.',
    notebook:'01_Clasificacion_Bancaria.ipynb',
    source:'https://archive.ics.uci.edu/dataset/222/bank+marketing',
    sourceName:'Bank Marketing · UCI',
    helper:'Elige un cliente real del conjunto de prueba. Estos casos incluyen aciertos y errores para ver cómo se comporta el modelo.',
    verdict:'Una probabilidad más alta indica mayor interés estimado. El umbral se eligió con validación; una predicción no garantiza que el cliente compre.'
  },
  energy: {
    title:'¿Cuánta calefacción necesita?', kicker:'REGRESIÓN · EDIFICIOS',
    intro:'Una red neuronal estima la carga de calefacción de un diseño de edificio a partir de ocho características físicas.',
    notebook:'02_Regresion_Energetica.ipynb',
    source:'https://archive.ics.uci.edu/dataset/242/energy+efficiency',
    sourceName:'Energy Efficiency · UCI',
    helper:'Elige una configuración del conjunto de prueba y compara la carga estimada con el valor real de la simulación.',
    verdict:'Los 768 diseños del dataset son simulados. El buen resultado dentro de estos datos no demuestra el mismo desempeño en edificios reales.'
  },
  forda: {
    title:'¿Qué patrón tiene esta señal?', kicker:'SERIES DE TIEMPO · AUTOMOCIÓN',
    intro:'Una CNN 1D analiza 500 mediciones consecutivas de ruido de motor y clasifica la señal en una de dos categorías.',
    notebook:'03_Series_Tiempo_FordA.ipynb',
    source:'https://www.timeseriesclassification.com/description.php?Dataset=FordA',
    sourceName:'FordA · UCR/UEA',
    helper:'Elige una señal de prueba para ver sus 500 mediciones y la categoría asignada por el modelo.',
    verdict:'FordA no identifica de forma pública y fiable cuál etiqueta es «normal» y cuál «falla». Por eso se muestran como clase 0 y clase 1.'
  }
};
const cache = {};
let active = 'bank';
let requested = 'bank';
let caseIndex = 0;
const panel = document.querySelector('#model-panel');
const fmt = (v, digits=3) => Number(v).toFixed(digits);
const pct = v => `${(Number(v)*100).toFixed(1)} %`;
const escapeHtml = value => String(value).replace(/[&<>"']/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const metric = (value, label) => `<div class="metric"><strong>${escapeHtml(value)}</strong><small>${escapeHtml(label)}</small></div>`;
const svgLine = (values, color='#1b9a72', min=null, max=null, width=520, height=190) => {
  const a = min ?? Math.min(...values), b = max ?? Math.max(...values);
  const range = b-a || 1;
  const pts=values.map((v,i)=>`${(i/(values.length-1||1)*width).toFixed(1)},${(height-((v-a)/range)*height).toFixed(1)}`).join(' ');
  return `<svg viewBox="0 0 ${width} ${height+18}" role="img" aria-label="Gráfica de datos"><line x1="0" y1="${height}" x2="${width}" y2="${height}" stroke="#c9d7cd"/><polyline fill="none" stroke="${color}" stroke-width="2.4" points="${pts}"/><text x="1" y="${height+15}" font-size="10" fill="#708078">Inicio</text><text x="${width-24}" y="${height+15}" font-size="10" fill="#708078">Fin</text></svg>`;
};
const confusion = c => `<div class="confusion" aria-label="Matriz de confusión">${[
  [c[0][0],'0 real / 0 predicho'],[c[0][1],'0 real / 1 predicho'],
  [c[1][0],'1 real / 0 predicho'],[c[1][1],'1 real / 1 predicho']
].map(([n,label])=>`<div><strong>${n.toLocaleString('es-MX')}</strong><span>${label}</span></div>`).join('')}</div>`;
const labels = {
  age:'Edad',job:'Ocupación',marital:'Estado civil',education:'Educación',default:'Impago',balance:'Saldo anual',housing:'Crédito vivienda',loan:'Préstamo personal',contact:'Contacto',day:'Día',month:'Mes',campaign:'Contactos campaña',pdays:'Días contacto previo',previous:'Contactos previos',poutcome:'Resultado previo',
  X1:'Compacidad relativa',X2:'Área superficial',X3:'Área de muros',X4:'Área de techo',X5:'Altura',X6:'Orientación',X7:'Área acristalada',X8:'Distribución del vidrio'
};
function metricsHtml(kind,d){const m=d.metrics;
  if(kind==='bank') return metric(fmt(m.roc_auc),'ROC-AUC · separación de clases')+metric(fmt(m.pr_auc),'PR-AUC · clase positiva')+metric(pct(m.recall),'Recall · clientes que sí contratan')+metric(fmt(m.f1),'F1 · clase positiva');
  if(kind==='energy') return metric(fmt(m.mae),'MAE · error medio')+metric(fmt(m.rmse),'RMSE · error cuadrático')+metric(fmt(m.r2,4),'R² · ajuste')+metric(fmt(m.baseline_mae),'MAE de referencia: media');
  return metric(pct(m.accuracy),'Exactitud en prueba')+metric(fmt(m.f1),'F1 · clase 1')+metric(pct(m.majority_accuracy),'Referencia: clase mayoritaria')+metric(`${d.n_test}`,'Señales en prueba');
}
function outcome(kind,e,d){
  if(kind==='energy')return `<div class="case-result"><div><span>Carga estimada</span><strong>${fmt(e.prediction,2)}</strong></div><div class="actual">Valor real<br><b>${fmt(e.actual,2)}</b></div></div>`;
  const name = x => kind==='bank' ? (x ? 'Sí contrata':'No contrata') : `Clase ${x}`;
  const p = pct(e.probability);
  return `<div class="case-result"><div><span>Predicción del modelo</span><strong>${name(e.prediction)}</strong></div><div class="actual">Valor real<br><b>${name(e.actual)}</b><br><small>Probabilidad de clase 1: ${p}</small></div></div>`;
}
function features(kind,e){
  if(kind==='forda') return `<div class="chart-box" style="margin:20px"><h5>500 mediciones consecutivas del motor</h5>${svgLine(e.signal)}</div>`;
  return `<div class="feature-grid">${Object.entries(e.features).map(([k,v])=>`<div><span>${escapeHtml(labels[k]||k)}</span><strong>${escapeHtml(typeof v==='number'?Number(v.toFixed(2)).toLocaleString('es-MX'):v)}</strong></div>`).join('')}</div>`;
}
function insight(kind,d){
  if(kind==='energy'){
    const points=d.scatter;
    const min=Math.min(...points.flat()),max=Math.max(...points.flat()),range=max-min||1;
    const dots=points.map(([a,b])=>`<circle cx="${(32+(a-min)/range*430).toFixed(1)}" cy="${(185-(b-min)/range*165).toFixed(1)}" r="2.1" fill="#1b9a72" fill-opacity=".65"/>`).join('');
    return `<p>Cuanto más cerca están los puntos de la diagonal, más próxima fue la estimación al valor real.</p><div class="chart-box"><h5>Valores reales frente a predichos · prueba</h5><svg viewBox="0 0 485 210" role="img" aria-label="Valores reales y predichos"><line x1="32" y1="185" x2="462" y2="20" stroke="#a3b4aa" stroke-dasharray="5 5"/>${dots}<text x="150" y="205" font-size="10" fill="#708078">Carga real →</text></svg></div><div class="explain"><b>Qué significa MAE ${fmt(d.metrics.mae)}</b>En promedio, la estimación se apartó ${fmt(d.metrics.mae)} unidades del valor simulado en los casos de prueba.</div>`;
  }
  return `<p>La matriz muestra aciertos en verde y errores en naranja. Cada número cuenta casos del conjunto de prueba.</p>${confusion(d.confusion)}<div class="explain"><b>${kind==='bank'?'Más que una exactitud global':'Comparado con adivinar siempre la clase mayoritaria'}</b>${kind==='bank' ? `Solo ${pct(d.metrics.prevalence)} de los clientes contratan. La PR-AUC de ${fmt(d.metrics.pr_auc)} supera la referencia de ${fmt(d.metrics.prevalence)}.` : `La red acierta ${pct(d.metrics.accuracy)} de las señales; elegir siempre la clase mayoritaria acertaría ${pct(d.metrics.majority_accuracy)}.`}</div>`;
}
function renderCase(){
  const d=cache[active], e=d.examples[caseIndex];
  const card=document.querySelector('#case-card');
  if(!card)return;
  card.innerHTML=outcome(active,e,d)+features(active,e);
  const count=document.querySelector('#case-count');
  if(count)count.textContent=`Caso ${caseIndex+1} de ${d.examples.length}`;
  const selector=document.querySelector('#case-select');
  if(selector)selector.value=String(caseIndex);
}
function render(kind){
  const d=cache[kind], def=defs[kind]; active=kind; caseIndex=0;
  document.querySelectorAll('.tab').forEach(tab=>{const selected=tab.dataset.model===kind;tab.classList.toggle('active',selected);tab.setAttribute('aria-selected',String(selected));tab.tabIndex=selected?0:-1;});
  panel.setAttribute('aria-labelledby',`tab-${kind}`);
  const options=d.examples.map((e,i)=>`<option value="${i}">Caso ${i+1} · ${kind==='energy'?`real ${fmt(e.actual,1)}`:e.actual===e.prediction?'acierto':'error'}</option>`).join('');
  panel.innerHTML=`<div class="model-wrap"><div class="model-intro"><div><div class="model-kicker">${def.kicker}</div><h3>${def.title}</h3><p>${def.intro}</p></div><a class="outline-link" href="https://colab.research.google.com/github/al073016/Practica-4-Modelos-de-machine-learning/blob/main/${def.notebook}" target="_blank" rel="noopener">Ver entrenamiento en Colab ↗</a></div>
    <div class="metric-grid">${metricsHtml(kind,d)}</div><div class="model-content"><div class="explore"><div class="panel-top"><h4>Prueba un ejemplo</h4><span class="tag">DATOS DE PRUEBA</span></div><p>${def.helper}</p><div class="case-controls"><select id="case-select" aria-label="Elegir caso de prueba">${options}</select><button id="next-case" type="button" aria-label="Mostrar siguiente caso">Siguiente →</button></div><div id="case-count" class="small-note"></div><div id="case-card" class="case-card"></div></div><div class="insight"><div class="panel-top"><h4>Cómo leer el resultado</h4></div>${insight(kind,d)}<p class="small-note">${def.verdict}</p><p class="small-note">Datos: <a href="${def.source}" target="_blank" rel="noopener"><u>${def.sourceName} ↗</u></a> · Entrenamiento ${d.n_train.toLocaleString('es-MX')} · Validación ${d.n_val.toLocaleString('es-MX')} · Prueba ${d.n_test.toLocaleString('es-MX')}.</p></div></div></div>`;
  document.querySelector('#case-select').addEventListener('change',ev=>{caseIndex=Number(ev.target.value);renderCase()});
  document.querySelector('#next-case').addEventListener('click',()=>{caseIndex=(caseIndex+1)%d.examples.length;renderCase()});
  renderCase();
}
async function load(kind){
  requested=kind;
  if(!cache[kind]){
    panel.innerHTML='<div class="loading">Cargando ejemplos y resultados…</div>';
    try{const response=await fetch(`data/${kind}.json`);if(!response.ok)throw new Error(`HTTP ${response.status}`);cache[kind]=await response.json()}
    catch(error){if(requested===kind)panel.innerHTML=`<div class="error">No se pudieron cargar los datos (${escapeHtml(error.message)}). Actualiza la página para intentarlo de nuevo.</div>`;return}
  }
  if(requested===kind)render(kind);
}
document.querySelectorAll('.tab').forEach(tab=>tab.addEventListener('click',()=>load(tab.dataset.model)));
load('bank');
