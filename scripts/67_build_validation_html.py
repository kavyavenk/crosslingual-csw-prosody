#!/usr/bin/env python3
"""
Build an interactive HTML sheet for manually validating the MaSaC discourse-marker
(DM) / repetition (Rep) detectors. For each sampled utterance it shows the
transcript, an audio player for the clip, and Yes/No controls for DM and Rep.
Progress auto-saves to the browser (localStorage); an "Export CSV" button writes
a file that scripts/66_score_dm_rep_validation.py can score directly.

Usage
-----
python3 scripts/67_build_validation_html.py
# then either double-click the HTML, or (more reliable audio) from repo root:
#   python3 -m http.server 8000
#   open http://localhost:8000/results/paper/validation/label_dm_rep.html
"""
import json
import pathlib
import pandas as pd

VAL_DIR = pathlib.Path("results/paper/validation")
BLANK = VAL_DIR / "masac_dm_rep_validation_BLANK.csv"
FEATURES = "results/functional_anchors/masac/masac_integrated_features.csv"
OUT = VAL_DIR / "label_dm_rep.html"
# HTML lives in results/paper/validation/ ; clips live in data/... at repo root.
REL_PREFIX = "../../../"


def main():
    blank = pd.read_csv(BLANK, dtype=str)
    feats = pd.read_csv(FEATURES, low_memory=False)[["utt_id", "wav"]]
    df = blank.merge(feats, on="utt_id", how="left")

    items = []
    for _, r in df.iterrows():
        items.append({
            "row": int(r["row"]),
            "utt_id": r["utt_id"],
            "lang": r["lang"],
            "condition": r["condition"],
            "text": r["text"],
            "audio": REL_PREFIX + str(r["wav"]),
        })

    data_json = json.dumps(items, ensure_ascii=False).replace("</", "<\\/")
    html = HTML_TEMPLATE.replace("__DATA__", data_json)
    OUT.write_text(html, encoding="utf-8")
    print(f"Wrote {len(items)} utterances -> {OUT}")
    print("Open it by double-clicking, or serve from repo root for reliable audio:")
    print("  python3 -m http.server 8000")
    print(f"  open http://localhost:8000/{OUT.as_posix()}")


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MaSaC DM / Repetition manual validation</title>
<style>
  :root { --bg:#0f1216; --card:#1a1f26; --ink:#e8edf2; --mut:#9aa7b3;
          --yes:#2e7d5b; --no:#7d2e3a; --acc:#3d7bd6; --line:#2a323c; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font:15px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; }
  header { position:sticky; top:0; z-index:5; background:#12161c;
           border-bottom:1px solid var(--line); padding:12px 18px; }
  h1 { font-size:16px; margin:0 0 6px; }
  .bar { display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
  .btn { background:var(--acc); color:#fff; border:0; border-radius:8px;
         padding:8px 14px; font-size:14px; cursor:pointer; }
  .btn.sec { background:#333c47; }
  .prog { color:var(--mut); font-size:13px; }
  .guide { color:var(--mut); font-size:12.5px; margin-top:8px; max-width:900px; }
  .wrap { max-width:900px; margin:18px auto; padding:0 14px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:12px;
          padding:14px 16px; margin:12px 0; }
  .card.done { border-color:#2f5d45; }
  .meta { display:flex; gap:8px; align-items:center; color:var(--mut);
          font-size:12px; margin-bottom:8px; flex-wrap:wrap; }
  .badge { background:#232b34; border-radius:20px; padding:2px 10px; }
  .txt { font-size:17px; margin:8px 0 12px; }
  audio { width:100%; margin-bottom:12px; }
  .q { display:flex; gap:10px; align-items:center; margin:6px 0; flex-wrap:wrap; }
  .q .lab { width:170px; color:var(--mut); font-size:13px; }
  .opt { border:1px solid var(--line); background:#151a20; color:var(--ink);
         border-radius:8px; padding:6px 14px; cursor:pointer; font-size:14px; }
  .opt.yes.on { background:var(--yes); border-color:var(--yes); color:#fff; }
  .opt.no.on  { background:var(--no);  border-color:var(--no);  color:#fff; }
  .notes { flex:1; min-width:180px; background:#151a20; color:var(--ink);
           border:1px solid var(--line); border-radius:8px; padding:6px 10px; }
  .filterbar { margin-left:auto; }
  select { background:#151a20; color:var(--ink); border:1px solid var(--line);
           border-radius:8px; padding:6px 10px; }
</style>
</head>
<body>
<header>
  <h1>MaSaC discourse-marker &amp; repetition &mdash; manual validation</h1>
  <div class="bar">
    <button class="btn" id="export">Export CSV</button>
    <button class="btn sec" id="importBtn">Import CSV</button>
    <input type="file" id="importFile" accept=".csv" style="display:none">
    <button class="btn sec" id="reset">Reset labels</button>
    <span class="prog" id="prog"></span>
    <span class="filterbar">
      <select id="filter">
        <option value="all">Show all</option>
        <option value="todo">Only unlabeled</option>
        <option value="done">Only labeled</option>
      </select>
    </span>
  </div>
  <div class="guide">
    <p style="margin:4px 0"><b>Discourse marker = Yes</b> &mdash; a
    filler/connector/reformulation/emphasis particle functioning pragmatically:
    EN <i>like, you know, so, actually, um, I mean</i>; HI <i>toh, matlab, arre,
    na, bilkul, yaar, acha, phir, lekin</i>. Judge the <i>function</i>, not just
    the string &rarr; mark <b>No</b> if a short marker only appears as a
    <i>substring</i> inside a bigger word (<i>par</i> in <i>pareshaan</i>,
    <i>aur</i> in <i>aurat</i>).</p>
    <p style="margin:4px 0"><b>Repetition = Yes</b> &mdash; a content
    word/short phrase repeated within this utterance for emphasis, agreement, or
    self-repair: <i>paisa hi paisa</i>, <i>roz roz</i>, <i>hai na, hai na</i>,
    <i>nahi nahi</i>, <i>main&hellip; main soch raha tha</i>. Mark <b>No</b> when
    nothing repeats, when only a <i>function word</i> recurs by normal grammar
    (<i>ki, hai, toh, the</i>), when a <i>proper name</i> repeats
    (<i>Monisha&hellip; Monisha</i> &rarr; excluded), or for echo words that are
    actually different (<i>chai-shai</i>).</p>
    <p style="margin:4px 0">Progress auto-saves in this browser; use the notes
    box for tricky cases, and click <b>Export CSV</b> when done.</p>
  </div>
</header>
<div class="wrap" id="list"></div>

<script>
const DATA = __DATA__;
const KEY = "masac_dm_rep_labels_v1";
let store = JSON.parse(localStorage.getItem(KEY) || "{}");

function save(){ localStorage.setItem(KEY, JSON.stringify(store)); updateProg(); }
function rec(id){ return store[id] || (store[id]={dm:null,rep:null,notes:""}); }

function updateProg(){
  let done=0;
  for(const d of DATA){ const r=store[d.utt_id];
    if(r && r.dm!==null && r.rep!==null) done++; }
  document.getElementById("prog").textContent = done+" / "+DATA.length+" labeled";
  document.querySelectorAll(".card").forEach(c=>{
    const r=store[c.dataset.id];
    c.classList.toggle("done", !!(r && r.dm!==null && r.rep!==null));
  });
}

function setOpt(id, field, val){
  rec(id)[field]=val; save();
  render();
}

function render(){
  const filter=document.getElementById("filter").value;
  const list=document.getElementById("list"); list.innerHTML="";
  for(const d of DATA){
    const r=store[d.utt_id]||{dm:null,rep:null,notes:""};
    const labeled = r.dm!==null && r.rep!==null;
    if(filter==="todo" && labeled) continue;
    if(filter==="done" && !labeled) continue;
    const card=document.createElement("div");
    card.className="card"+(labeled?" done":""); card.dataset.id=d.utt_id;
    card.innerHTML=`
      <div class="meta">
        <span class="badge">#${d.row}</span>
        <span class="badge">${d.lang}</span>
        <span class="badge">${d.condition}</span>
        <span class="badge">${d.utt_id}</span>
      </div>
      <div class="txt">${escapeHtml(d.text)}</div>
      <audio controls preload="none" src="${encodeURI(d.audio)}"></audio>
      <div class="q"><span class="lab">Discourse marker?</span>
        <button class="opt yes ${r.dm===1?'on':''}" data-f="dm" data-v="1">Yes</button>
        <button class="opt no ${r.dm===0?'on':''}"  data-f="dm" data-v="0">No</button>
      </div>
      <div class="q"><span class="lab">Repetition?</span>
        <button class="opt yes ${r.rep===1?'on':''}" data-f="rep" data-v="1">Yes</button>
        <button class="opt no ${r.rep===0?'on':''}"  data-f="rep" data-v="0">No</button>
        <input class="notes" placeholder="notes (optional)" value="${escapeHtml(r.notes||'')}">
      </div>`;
    card.querySelectorAll(".opt").forEach(b=>{
      b.onclick=()=>setOpt(d.utt_id, b.dataset.f, parseInt(b.dataset.v));
    });
    card.querySelector(".notes").onchange=e=>{ rec(d.utt_id).notes=e.target.value; save(); };
    list.appendChild(card);
  }
  updateProg();
}

function escapeHtml(s){ return String(s).replace(/[&<>"']/g,
  c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }

function csvCell(s){ s=String(s==null?"":s);
  return /[",\n]/.test(s) ? '"'+s.replace(/"/g,'""')+'"' : s; }

document.getElementById("export").onclick=()=>{
  const rows=[["row","utt_id","lang","condition","text","human_dm","human_rep","notes"]];
  for(const d of DATA){ const r=store[d.utt_id]||{};
    rows.push([d.row,d.utt_id,d.lang,d.condition,d.text,
      r.dm==null?"":r.dm, r.rep==null?"":r.rep, r.notes||""]); }
  const csv="\ufeff"+rows.map(r=>r.map(csvCell).join(",")).join("\n");
  const blob=new Blob([csv],{type:"text/csv;charset=utf-8"});
  const a=document.createElement("a"); a.href=URL.createObjectURL(blob);
  a.download="masac_dm_rep_validation_FILLED.csv"; a.click();
};

document.getElementById("importBtn").onclick=()=>document.getElementById("importFile").click();
document.getElementById("importFile").onchange=(e)=>{
  const f=e.target.files[0]; if(!f) return;
  const rd=new FileReader();
  rd.onload=()=>{ parseCsv(rd.result); save(); render(); };
  rd.readAsText(f);
};
function parseCsv(text){
  const lines=text.replace(/\ufeff/,"").split(/\r?\n/).filter(x=>x.length);
  const head=splitCsv(lines[0]);
  const iId=head.indexOf("utt_id"), iDm=head.indexOf("human_dm"),
        iRep=head.indexOf("human_rep"), iN=head.indexOf("notes");
  for(let k=1;k<lines.length;k++){ const c=splitCsv(lines[k]); const id=c[iId];
    if(!id) continue;
    const r=rec(id);
    r.dm = c[iDm]===""||c[iDm]==null ? null : parseInt(c[iDm]);
    r.rep= c[iRep]===""||c[iRep]==null? null : parseInt(c[iRep]);
    r.notes = iN>=0 ? (c[iN]||"") : r.notes;
  }
}
function splitCsv(line){ const out=[]; let cur="",q=false;
  for(let i=0;i<line.length;i++){ const ch=line[i];
    if(q){ if(ch=='"'){ if(line[i+1]=='"'){cur+='"';i++;} else q=false; } else cur+=ch; }
    else { if(ch=='"')q=true; else if(ch==',' ){out.push(cur);cur="";} else cur+=ch; } }
  out.push(cur); return out; }

document.getElementById("reset").onclick=()=>{
  if(confirm("Clear all labels in this browser?")){ store={}; save(); render(); } };
document.getElementById("filter").onchange=render;

render();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
