/* Minimal, dependency-free leaderboard UI:
 * - loads ../leaderboard/leaderboard.csv
 * - search + filters (model, date)
 * - sortable columns
 * - column toggles
 */
function parseCSV(text){
  const lines = text.trim().split(/\r?\n/);
  const header = lines[0].split(",");
  const rows = [];
  for(let i=1;i<lines.length;i++){
    if(!lines[i].trim()) continue;
    const cols = [];
    // naive CSV parse with quotes
    let cur="", inQ=false;
    for(let j=0;j<lines[i].length;j++){
      const ch = lines[i][j];
      if(ch === '"'){ inQ = !inQ; continue; }
      if(ch === "," && !inQ){ cols.push(cur); cur=""; continue; }
      cur += ch;
    }
    cols.push(cur);
    const obj = {};
    header.forEach((h, idx) => obj[h] = (cols[idx] ?? "").trim());
    rows.push(obj);
  }
  return rows;
}

function daysAgo(dateStr){
  const d = new Date(dateStr);
  if(isNaN(d.getTime())) return Infinity;
  const now = new Date();
  return (now - d) / (1000*60*60*24);
}

const state = {
  rows: [],
  filtered: [],
  sortKey: "score",
  sortDir: "desc", // asc|desc
  hiddenCols: new Set(),
};

function renderTable(){
  const tbody = document.querySelector("#tbl tbody");
  tbody.innerHTML = "";
  const rows = state.filtered;

  rows.forEach((r, idx) => {
    const tr = document.createElement("tr");
    const rank = idx + 1;
    const cells = [
      ["rank", rank],
      ["team", r.team],
      ["model", r.model],
      ["score", r.score],
      ["timestamp_utc", r.timestamp_utc],
      ["notes", r.notes || ""],
    ];
    cells.forEach(([k, v]) => {
      const td = document.createElement("td");
      td.dataset.key = k;
      td.textContent = v;
      if(k === "rank") td.classList.add("rank");
      if(k === "score") td.classList.add("score");
      if(state.hiddenCols.has(k)) td.style.display = "none";
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });

  // hide headers accordingly
  document.querySelectorAll("#tbl thead th").forEach(th => {
    const k = th.dataset.key;
    th.style.display = state.hiddenCols.has(k) ? "none" : "";
  });

  document.getElementById("status").textContent =
    rows.length ? `${rows.length} result(s)` : "No results";
}

function applyFilters(){
  const q = document.getElementById("search").value.toLowerCase().trim();
  const model = document.getElementById("modelFilter").value;
  const date = document.getElementById("dateFilter").value;

  let rows = [...state.rows];

  if(model !== "all"){
    rows = rows.filter(r => (r.model || "").toLowerCase() === model);
  }

  if(date !== "all"){
    const maxDays = (date === "last30") ? 30 : 180;
    rows = rows.filter(r => daysAgo(r.timestamp_utc) <= maxDays);
  }

  if(q){
    rows = rows.filter(r => {
      const hay = `${r.team} ${r.model} ${r.notes} ${r.timestamp_utc}`.toLowerCase();
      return hay.includes(q);
    });
  }

  // Sort
  const k = state.sortKey;
  const dir = state.sortDir === "asc" ? 1 : -1;
  rows.sort((a,b) => {
    let av = a[k], bv = b[k];
    if(k === "score"){
      av = parseFloat(av); bv = parseFloat(bv);
      if(isNaN(av)) av = -Infinity;
      if(isNaN(bv)) bv = -Infinity;
      return (av - bv) * dir;
    }
    // default string
    av = (av ?? "").toString().toLowerCase();
    bv = (bv ?? "").toString().toLowerCase();
    if(av < bv) return -1 * dir;
    if(av > bv) return  1 * dir;
    return 0;
  });

  state.filtered = rows;
  renderTable();
}

function setupColumnToggles(){
  const cols = [
    ["rank","Rank"],
    ["team","Team"],
    ["model","Model"],
    ["score","Score"],
    ["timestamp_utc","Date (UTC)"],
    ["notes","Notes"],
  ];
  const wrap = document.getElementById("columnToggles");
  wrap.innerHTML = "";
  cols.forEach(([k,label]) => {
    const id = `col_${k}`;
    const lab = document.createElement("label");
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = !state.hiddenCols.has(k);
    cb.id = id;
    cb.addEventListener("change", () => {
      if(cb.checked) state.hiddenCols.delete(k);
      else state.hiddenCols.add(k);
      renderTable();
    });
    lab.appendChild(cb);
    const sp = document.createElement("span");
    sp.textContent = label;
    lab.appendChild(sp);
    wrap.appendChild(lab);
  });
}

function setupSorting(){
  document.querySelectorAll("#tbl thead th").forEach(th => {
    th.addEventListener("click", () => {
      const k = th.dataset.key;
      if(!k) return;
      if(state.sortKey === k){
        state.sortDir = (state.sortDir === "asc") ? "desc" : "asc";
      }else{
        state.sortKey = k;
        state.sortDir = (k === "score") ? "desc" : "asc";
      }
      applyFilters();
    });
  });
}

async function main(){
  const status = document.getElementById("status");
  try{
    const res = await fetch("../leaderboard/leaderboard.csv", {cache:"no-store"});
    const txt = await res.text();
    const rows = parseCSV(txt);

    // normalize and compute rank order by score (descending) initially
    const cleaned = rows
      .filter(r => r.team) // ignore empty template line
      // .map(r => ({
      //   timestamp_utc: r.timestamp_utc,
      //   team: r.team,
      //   model: (r.model || "").toLowerCase(),
      //   score: r.score,
      //   notes: r.notes || "",
      // }));
      .map(r => ({
        timestamp_utc: r.timestamp,
        team: r.team_name,
        model: (r.model || "").toLowerCase(),
        score: r.validation_accuracy,
        notes: r.validation_f1_score || "",
      }));

    state.rows = cleaned;

    // fill model options
    const modelSet = new Set(cleaned.map(r => r.model).filter(Boolean));
    const sel = document.getElementById("modelFilter");
    [...modelSet].sort().forEach(m => {
      const opt = document.createElement("option");
      opt.value = m;
      opt.textContent = m;
      sel.appendChild(opt);
    });

    setupColumnToggles();
    setupSorting();

    document.getElementById("search").addEventListener("input", applyFilters);
    document.getElementById("modelFilter").addEventListener("change", applyFilters);
    document.getElementById("dateFilter").addEventListener("change", applyFilters);

    // default: sort by score desc
    state.sortKey = "score";
    state.sortDir = "desc";
    applyFilters();
  }catch(e){
    status.textContent = "Failed to load leaderboard.";
    console.error(e);
  }
}

main();
