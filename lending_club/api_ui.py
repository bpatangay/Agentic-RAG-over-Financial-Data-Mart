from __future__ import annotations

from .rag import SUPPORTED

EXTRA_SAMPLES = (
    "Which FICO bands have the highest charge-off rate?",
    "What is the total loan amount charged off by loan grade in 2017?",
    "Show the quarterly trend of fully paid and charged-off loans.",
    "What is the loss given default (LGD) on charged off loans by year?",
    "What is the recovery rate on charged off loans by grade?",
)

QUERY_UI = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Lending Club Financial Data Mart</title>
  <style>
    :root { font-family: Inter, ui-sans-serif, system-ui, sans-serif; color:#16243a; background:#f4f7fb; }
    * { box-sizing:border-box } body { margin:0 } main { max-width:1120px; margin:auto; padding:28px 20px 64px }
    header { display:flex; justify-content:space-between; align-items:center; margin-bottom:24px }
    .brand { font-size:clamp(1.75rem,3.5vw,2.4rem); font-weight:800; line-height:1 } .health { color:#60718a; font-size:.82rem }
    .hero { margin-bottom:20px } .hero h1 { font-size:clamp(1.6rem,3.5vw,2.15rem); line-height:1.05; letter-spacing:-.04em; margin:0 }
    .lede { color:#60718a; max-width:720px; line-height:1.6 }
    .panel { background:#fff; border:1px solid #dce5ef; border-radius:18px; box-shadow:0 16px 45px rgba(11,37,69,.08); padding:22px; margin-bottom:18px }
    form { display:grid; gap:10px } form label { font-size:1rem; font-weight:800 }
    textarea { width:100%; min-height:92px; resize:vertical; border:1px solid #b9c9dc; border-radius:12px; padding:14px; font:inherit }
    textarea:focus { outline:3px solid #dbeaff; border-color:#1769e0 }
    .actions { display:flex; justify-content:space-between; align-items:center; gap:12px }
    button { border:0; border-radius:9px; background:#1769e0; color:#fff; cursor:pointer; font-weight:800; min-height:42px; padding:10px 16px }
    button.secondary { background:#edf3fa; color:#26415e } button:disabled { opacity:.65; cursor:wait }
    .samples { display:flex; flex-wrap:wrap; gap:8px; margin-top:14px }
    .samples button { background:#edf3fa; color:#26415e; font-size:.76rem; min-height:32px; padding:6px 10px }
    #status { display:none; margin-top:12px; border-radius:10px; padding:10px 12px; font-size:.85rem }
    #status.visible { display:block } #status.loading { background:#eef5ff; color:#0e4fae } #status.error { background:#fff3f1; color:#b42318 }
    #status.success { background:#edf9f5; color:#0d8065 }
    .results { display:grid; gap:14px; grid-template-columns:minmax(0,1fr) 310px; margin-top:14px }
    .hidden { display:none!important } .card { background:#fff; border:1px solid #dce5ef; border-radius:16px; padding:18px }
    .answer { border-left:4px solid #23ad83; background:#f5faf8; border-radius:0 10px 10px 0; padding:12px 14px; margin:12px 0 16px }
    .section-title { font-size:1rem; line-height:1.2; margin:0 0 8px }
    #summary { font-size:.95rem; color:#16243a; line-height:1.5; }
    .meta-badges { display:flex; flex-wrap:wrap; gap:6px; margin:10px 0 }
    .badge { background:#edf3fa; color:#26415e; border-radius:999px; padding:4px 10px; font-size:.74rem; font-weight:700 }
    .sql-box { background:#081b33; color:#d9e7fa; border-radius:10px; padding:12px; font-size:.78rem; overflow:auto; white-space:pre-wrap; margin-top:10px }
    .table-wrap { overflow:auto; margin-top:14px } table { border-collapse:collapse; width:100%; font-size:.86rem }
    th,td { border-bottom:1px solid #e5ebf3; padding:9px 10px; text-align:left } th { color:#60718a; background:#f8fafc }
    .metric-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(120px,1fr)); gap:8px; margin-bottom:14px }
    .metric { background:#f4f7fb; border-radius:10px; padding:10px } .metric strong { display:block; font-size:1.05rem } .metric span { color:#60718a; font-size:.72rem }
    .chart { display:flex; align-items:end; gap:8px; height:150px; padding-top:12px; margin-bottom:24px }
    .bar { flex:1; min-width:24px; background:#1769e0; border-radius:5px 5px 0 0; position:relative }
    .bar span { color:#60718a; font-size:.66rem; position:absolute; bottom:-20px; left:50%; transform:translateX(-50%) }
    details { grid-column:1/-1; margin-top:10px } details pre { background:#081b33; color:#d9e7fa; border-radius:10px; padding:14px; overflow:auto; white-space:pre-wrap; font-size:.78rem }
    .empty { color:#60718a; margin-top:14px }
    @media (max-width:760px) { .results { grid-template-columns:1fr } .actions { align-items:stretch; flex-direction:column } button { width:100% } }
  </style>
</head>
<body>
<main>
  <header><div class="brand" data-testid="data-mart-title">Lending Club Data Mart</div><div class="health" id="health">Checking API…</div></header>
  <section class="hero"><h1 data-testid="hero-title">Ask the portfolio<br>what happened.</h1><p class="lede">Explore 2015–2018 Lending Club loan performance with schema-grounded Agentic RAG SQL, decision-tree risk rules, and LGD recovery analytics.</p></section>
  <section class="panel">
    <form id="query-form">
      <label for="question" class="section-title" data-testid="analysis-question">Analysis question</label>
      <textarea id="question" placeholder="What is the average loan amount by grade?" required></textarea>
      <div class="actions"><span style="color:#60718a;font-size:.78rem">Read-only validated SQL over indexed star schema with LRU caching.</span><button id="submit" type="submit">Run analysis</button></div>
    </form>
    <div class="samples" id="samples" aria-label="Sample questions"></div>
    <div id="status" role="status" aria-live="polite"></div>
  </section>
  <p id="empty" class="empty"><strong>Ready for analysis.</strong> Run a sample question or write your own.</p>
  <div id="results" class="results hidden">
    <section class="card">
      <h2 class="section-title" data-testid="analysis-result">Analysis result</h2>
      <div id="answer" class="answer"></div>
      <div id="metrics" class="metric-grid"></div>
      <div id="chart" class="chart"></div>
      <div class="table-wrap"><table><thead id="thead"></thead><tbody id="tbody"></tbody></table></div>
    </section>
    <aside class="card">
      <h3 class="section-title" data-testid="result-summary">Result summary</h3>
      <div id="summary"></div>
      <div id="badges" class="meta-badges"></div>
      <pre id="sql-preview" class="sql-box"></pre>
    </aside>
    <details><summary>Complete JSON response</summary><pre id="json"></pre></details>
  </div>
  <section class="panel" style="margin-top:20px">
    <details id="rules-details"><summary><strong>Auditable Decision Tree Risk Rules</strong> (click to load)</summary><pre id="rules-text">Loading rules…</pre></details>
  </section>
</main>
<script>
const samples = """ + repr([*SUPPORTED, *EXTRA_SAMPLES]).replace("'", '"') + """;
const form = document.getElementById("query-form");
const question = document.getElementById("question");
const button = document.getElementById("submit");
const status = document.getElementById("status");
const empty = document.getElementById("empty");
const results = document.getElementById("results");

function setStatus(kind, text) { status.className = "visible " + kind; status.textContent = text; }
samples.forEach((text) => { const b=document.createElement("button"); b.type="button"; b.textContent=text; b.onclick=()=>{question.value=text; question.focus();}; document.getElementById("samples").appendChild(b); });

fetch("/health").then(r=>r.json()).then(p=>{ document.getElementById("health").textContent=p.status==="ok"?`API ready · Models: ${p.models_loaded?"loaded":"unloaded"}`:"Database not loaded"; }).catch(()=>document.getElementById("health").textContent="API unavailable");

document.getElementById("rules-details").addEventListener("toggle", (e) => {
  if (e.target.open) {
    fetch("/rules").then(r=>r.json()).then(d=>{ document.getElementById("rules-text").textContent = d.rules || "No rules available."; }).catch(()=>{ document.getElementById("rules-text").textContent = "Rules unavailable."; });
  }
});

function formatValue(value) {
  if (typeof value !== "number") return value == null ? "—" : String(value);
  if (Number.isInteger(value)) return value.toLocaleString();
  return value.toLocaleString(undefined,{maximumFractionDigits:2});
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const value=question.value.trim();
  if (!value) { setStatus("error","Please enter a question."); question.focus(); return; }
  button.disabled=true; button.textContent="Running analysis…"; setStatus("loading","Translating the question and executing validated read-only SQL…");
  empty.classList.add("hidden"); results.classList.add("hidden");
  try {
    const response=await fetch("/query",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({question:value})});
    const payload=await response.json();
    if (!response.ok) throw new Error(payload.detail||"The analysis could not be completed.");
    document.getElementById("answer").textContent=payload.answer||payload.summary||"Query completed.";
    document.getElementById("summary").textContent=payload.summary || `${(payload.rows||[]).length} rows returned.`;
    document.getElementById("badges").innerHTML = `<span class="badge">Mode: ${payload.execution_mode||"unknown"}</span><span class="badge">Latency: ${payload.latency_ms ?? "—"} ms</span><span class="badge">Rows: ${(payload.rows||[]).length}</span>`;
    document.getElementById("sql-preview").textContent = payload.sql || "-- No SQL executed --";
    document.getElementById("json").textContent=JSON.stringify(payload,null,2);
    const columns=payload.columns&&payload.columns.length?payload.columns:Object.keys(payload.rows?.[0]||{});
    document.getElementById("thead").innerHTML="<tr>"+columns.map(c=>`<th>${c.replaceAll("_"," ")}</th>`).join("")+"</tr>";
    document.getElementById("tbody").innerHTML=(payload.rows||[]).map(r=>"<tr>"+columns.map(c=>`<td>${formatValue(r[c])}</td>`).join("")+"</tr>").join("")||(columns.length?`<tr><td colspan="${columns.length}">No matching rows.</td></tr>`:"");
    const metrics=document.getElementById("metrics"); metrics.innerHTML="";
    (payload.rows||[]).slice(0,6).forEach((row)=>{
      const numericKey=columns.find(c=>typeof row[c]=="number" && !["year","month","quarter"].includes(c));
      if (numericKey) { const m=document.createElement("div"); m.className="metric"; m.innerHTML=`<strong>${formatValue(row[numericKey])}</strong><span>${row[columns[0]]}</span>`; metrics.appendChild(m); }
    });
    const chart=document.getElementById("chart"); chart.innerHTML="";
    const numericKey=(payload.rows||[])[0]&&columns.find(c=>typeof payload.rows[0][c]=="number" && !["year","month","quarter"].includes(c));
    if (numericKey) {
      const vals=(payload.rows||[]).map(r=>Number(r[numericKey])||0), max=Math.max(...vals,1);
      vals.forEach((v,i)=>{const bar=document.createElement("div"); bar.className="bar"; bar.style.height=`${Math.max(4,(v/max)*100)}%`; bar.title=`${payload.rows[i][columns[0]]}: ${formatValue(v)}`; bar.innerHTML=`<span>${String(payload.rows[i][columns[0]]).slice(0,8)}</span>`; chart.appendChild(bar);});
    }
    results.classList.remove("hidden"); setStatus("success","Analysis complete.");
  } catch (error) {
    setStatus("error",error.message||"The query could not be completed.");
  } finally { button.disabled=false; button.textContent="Run analysis"; }
});
</script>
</body>
</html>"""
