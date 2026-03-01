function parseCSV(text) {
  const lines = text.trim().split(/\r?\n/);
  if (lines.length < 2) return [];
  const header = lines[0].split(",");
  const rows = [];
  for (let i = 1; i < lines.length; i++) {
    if (!lines[i].trim()) continue;
    const cols = [];
    let cur = "", inQ = false;
    for (let j = 0; j < lines[i].length; j++) {
      const ch = lines[i][j];
      if (ch === '"') { inQ = !inQ; continue; }
      if (ch === "," && !inQ) { cols.push(cur); cur = ""; continue; }
      cur += ch;
    }
    cols.push(cur);
    const obj = {};
    header.forEach((h, idx) => obj[h.trim()] = (cols[idx] ?? "").trim());
    rows.push(obj);
  }
  return rows;
}

function daysAgo(dateStr) {
  const d = new Date(dateStr);
  if (isNaN(d.getTime())) return Infinity;
  return (new Date() - d) / (1000 * 60 * 60 * 24);
}

const state = {
  rows: [],
  filtered: [],
  sortKey: "macro_f1",
  sortDir: "desc",
  hiddenCols: new Set(),
};

function renderTable() {
  const tbody = document.querySelector("#tbl tbody");
  tbody.innerHTML = "";
  const rows = state.filtered;

  let lastScore = null;
  let lastRank = 0;

  rows.forEach((r, idx) => {
    const tr = document.createElement("tr");
    let rank;

    if (state.sortKey === "macro_f1") {
      const currentScore = parseFloat(r.macro_f1);
      if (currentScore === lastScore) {
        rank = lastRank;
      } else {
        rank = idx + 1;
        lastRank = rank;
        lastScore = currentScore;
      }
    } else {
      rank = idx + 1;
    }

    // Mapping strictly to the HTML header order
    const cells = [
      ["rank", rank],
      ["team", r.team],
      ["type", r.type],
      ["model", r.model],
      ["macro_f1", r.macro_f1],
      ["accuracy", r.accuracy],
      ["precision", r.precision],
      ["recall", r.recall],
      ["timestamp_utc", r.timestamp_utc],
    ];

    cells.forEach(([k, v]) => {
      const td = document.createElement("td");
      td.dataset.key = k;

      if (k === "rank") {
        let medal = "";
        if (v === 1) medal = " 🥇";
        else if (v === 2) medal = " 🥈";
        else if (v === 3) medal = " 🥉";
        td.innerHTML = `<strong>${v}${medal}</strong>`;
        td.className = `rank-${v}`;
      }
      else if (["macro_f1", "accuracy", "precision", "recall"].includes(k)) {
        const num = parseFloat(v);
        // Multiply by 100 and fix to 2 decimal places
        td.textContent = isNaN(num) ? v : (num * 100).toFixed(2) + "%";
      }
      else {
        // Direct value from CSV (Team, Type, Model, Date) - NO ICONS
        td.textContent = v;
      }

      if (state.hiddenCols.has(k)) td.style.display = "none";
      tr.appendChild(td);
    });

    tbody.appendChild(tr);
  });

  // Sync Header Hiding
  document.querySelectorAll("#tbl thead th").forEach(th => {
    const k = th.dataset.key;
    th.style.display = state.hiddenCols.has(k) ? "none" : "";
  });

  document.getElementById("status").textContent = rows.length + " result(s)";
}

function applyFilters() {
  const q = document.getElementById("search").value.toLowerCase().trim();
  const model = document.getElementById("modelFilter").value;
  const date = document.getElementById("dateFilter").value;

  let rows = [...state.rows];

  if (model !== "all") rows = rows.filter(r => r.model.toLowerCase() === model);
  if (date !== "all") {
    const limit = (date === "last30") ? 30 : 180;
    rows = rows.filter(r => daysAgo(r.timestamp_utc) <= limit);
  }
  if (q) {
    rows = rows.filter(r =>
      `${r.team} ${r.model} ${r.type}`.toLowerCase().includes(q)
    );
  }

  const k = state.sortKey;
  const dir = state.sortDir === "asc" ? 1 : -1;
  rows.sort((a, b) => {
    let av = a[k], bv = b[k];
    if (["macro_f1", "accuracy", "precision", "recall"].includes(k)) {
      av = parseFloat(av) || -1;
      bv = parseFloat(bv) || -1;
      return (av - bv) * dir;
    }
    return (av.toString().toLowerCase() < bv.toString().toLowerCase() ? -1 : 1) * dir;
  });

  state.filtered = rows;
  renderTable();
}

function setupColumnToggles() {
  const cols = [
    ["rank", "Rank"], ["team", "Team"], ["type", "Type"], ["model", "Model"],
    ["macro_f1", "Macro-F1"], ["accuracy", "Accuracy"], ["precision", "Precision"],
    ["recall", "Recall"], ["timestamp_utc", "Date"]
  ];
  const wrap = document.getElementById("columnToggles");
  wrap.innerHTML = "";
  cols.forEach(([k, label]) => {
    const lab = document.createElement("label");
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = !state.hiddenCols.has(k);
    cb.addEventListener("change", () => {
      if (cb.checked) state.hiddenCols.delete(k);
      else state.hiddenCols.add(k);
      renderTable();
    });
    lab.append(cb, label);
    wrap.appendChild(lab);
  });
}

function setupSorting() {
  document.querySelectorAll("#tbl thead th").forEach(th => {
    th.addEventListener("click", () => {
      const k = th.dataset.key;
      if (!k) return;
      state.sortDir = (state.sortKey === k && state.sortDir === "desc") ? "asc" : "desc";
      state.sortKey = k;
      applyFilters();
    });
  });
}

async function main() {
  try {
    const res = await fetch("/Graph4ASD-Challenge/leaderboard/leaderboard.csv", { cache: "no-store" });
    const txt = await res.text();
    state.rows = parseCSV(txt);

    const modelSet = new Set(state.rows.map(r => r.model).filter(Boolean));
    const sel = document.getElementById("modelFilter");
    [...modelSet].sort().forEach(m => {
      const opt = document.createElement("option");
      opt.value = m.toLowerCase(); opt.textContent = m;
      sel.appendChild(opt);
    });

    setupColumnToggles();
    setupSorting();
    applyFilters();

    document.getElementById("search").addEventListener("input", applyFilters);
    document.getElementById("modelFilter").addEventListener("change", applyFilters);
    document.getElementById("dateFilter").addEventListener("change", applyFilters);
  } catch (e) {
    console.error(e);
  }
}

main();
