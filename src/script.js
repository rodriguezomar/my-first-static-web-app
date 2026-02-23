/* ====== Navegación de tabs ====== */
document.querySelectorAll('.tab-btn').forEach(btn=>{
  btn.addEventListener('click', ()=>{
    document.querySelectorAll('.tab-btn').forEach(b=>b.classList.remove('active'));
    document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('tab-'+btn.dataset.tab).classList.add('active');
  });
});

/* ====== Guardar datos generales ====== */
let generalData = {};
document.getElementById('saveGeneral').addEventListener('click', ()=>{
  generalData = {
    informe: document.getElementById('genInforme').value || '',
    fecha: document.getElementById('genFecha').value || '',
    celda: document.getElementById('genCelda').value || '',
    reactor: document.getElementById('genReactor').value || '',
    barra: document.getElementById('genBarra').value || '',
    vmax: parseFloat(document.getElementById('genVmax').value || '0'),
    imax: parseFloat(document.getElementById('genImax').value || '0'),
    tmax: parseFloat(document.getElementById('genTmax').value || '0'),
    tiempo: parseFloat(document.getElementById('genTiempo').value || '0')
  };
  document.getElementById('generalSaved').textContent = 'Datos guardados.';
});

/* ====== Utilidades comunes ====== */
const sum = arr => arr.reduce((acc,v)=>acc+v,0);
const mean = arr => (arr.length? sum(arr)/arr.length : 0);
const trapz = (x,y)=>{let s=0; for(let i=1;i<x.length;i++) s+=0.5*(y[i]+y[i-1])*(x[i]-x[i-1]); return s;};
const mse = (a,b)=>{ let s=0; for(let i=0;i<a.length;i++){ const d=a[i]-b[i]; s+=d*d; } return s/a.length; };

/* Canvas helpers */
function setupCanvas(canvas){
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr,0,0,dpr,0,0);
  return { ctx, dpr, W: canvas.width, H: canvas.height };
}
function plotLines(canvas, series, opts={}){
  const { ctx, dpr, W, H } = setupCanvas(canvas);
  ctx.clearRect(0,0,W,H);
  const allX = series.flatMap(s=>s.x), allY = series.flatMap(s=>s.y);
  if (!allX.length) return;
  const minX=Math.min(...allX), maxX=Math.max(...allX);
  const minY=Math.min(...allY), maxY=Math.max(...allY);
  const padL=40, padR=10, padT=10, padB=20;
  const x2px=v=>padL+(v-minX)/(maxX-minX)*(W/dpr-padL-padR);
  const y2px=v=>H/dpr-padB-(v-minY)/(maxY-minY)*(H/dpr-padT-padB);
  ctx.strokeStyle='#203040'; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(padL, padT); ctx.lineTo(padL, H/dpr-padB); ctx.lineTo(W/dpr-padR, H/dpr-padB); ctx.stroke();
  for (const s of series){
    ctx.strokeStyle = s.color || '#56b6c2'; ctx.lineWidth = s.width || 1.5;
    ctx.beginPath();
    for (let i=0;i<s.x.length;i++){
      const X=x2px(s.x[i]), Y=y2px(s.y[i]);
      if (i===0) ctx.moveTo(X,Y); else ctx.lineTo(X,Y);
    }
    ctx.stroke();
  }
  if (opts.points){
    ctx.fillStyle = opts.pointColor || '#e66';
    for (const p of opts.points){ const X=x2px(p.x), Y=y2px(p.y); ctx.beginPath(); ctx.arc(X,Y,3,0,2*Math.PI); ctx.fill(); }
  }
}

/* ====== Savitzky–Golay (LS centrado) ====== */
function savgol(y, windowSize=5, polyOrder=1){
  if (windowSize%2===0) windowSize++;
  const half = Math.floor(windowSize/2);
  const res = new Array(y.length).fill(0);
  const X = []; for (let i=-half;i<=half;i++){ const row=[]; for (let p=0;p<=polyOrder;p++) row.push(Math.pow(i,p)); X.push(row); }
  const XT = transpose(X), XT_X = matmul(XT,X), XT_X_inv = invertSymmetric(XT_X), pinv = matmul(XT_X_inv, XT);
  const e = Array.from({length:polyOrder+1}, (_,i)=> i===0?1:0);
  const w = matmul([e], pinv)[0];
  for (let i=0;i<y.length;i++){
    let acc=0, wsum=0;
    for (let k=-half;k<=half;k++){
      const j=i+k;
      const yy=(j<0||j>=y.length)? y[Math.max(0,Math.min(y.length-1,j))] : y[j];
      const wk = w[k+half];
      acc += wk*yy; wsum += wk;
    }
    res[i] = acc/(wsum||1);
  }
  return res;
  function matmul(A,B){ const r=A.length,c=B[0].length,m=B.length;
    const R=Array.from({length:r},()=>Array(c).fill(0));
    for(let i=0;i<r;i++) for(let j=0;j<c;j++){ let s=0; for(let k=0;k<m;k++) s+=A[i][k]*B[k][j]; R[i][j]=s; }
    return R;}
  function transpose(A){ const r=A.length,c=A[0].length;
    const T=Array.from({length:c},()=>Array(r).fill(0));
    for(let i=0;i<r;i++) for(let j=0;j<c;j++) T[j][i]=A[i][j]; return T;}
  function invertSymmetric(M){
    const n=M.length; const A=M.map(row=>row.slice()); const I=Array.from({length:n},(_,i)=>Array.from({length:n},(_,j)=>i===j?1:0));
    for(let i=0;i<n;i++){
      let piv=i; for(let r=i;r<n;r++) if (Math.abs(A[r][i])>Math.abs(A[piv][i])) piv=r;
      if(piv!==i){ [A[i],A[piv]]=[A[piv],A[i]]; [I[i],I[piv]]=[I[piv],I[i]]; }
      const v=A[i][i]; if(Math.abs(v)<1e-12) throw new Error("Singular SG");
      for(let j=0;j<n;j++){ A[i][j]/=v; I[i][j]/=v; }
      for(let r=0;r<n;r++){ if(r===i) continue; const f=A[r][i];
        for(let j=0;j<n;j++){ A[r][j]-=f*A[i][j]; I[r][j]-=f*I[i][j]; }
      }
    }
    return I;
  }
}

/* ====== Polinomio LS e iteración baseline ====== */
function polyfitLS(x,y,order=2){
  const n=order+1;
  const X = x.map(v => Array.from({length:n},(_,p)=>Math.pow(v,p)));
  const XT=transpose(X), XT_X=matmul(XT,X), XT_y=matvec(XT,y);
  return solveGauss(XT_X, XT_y);
  function transpose(A){ const r=A.length,c=A[0].length; const T=Array.from({length:c},()=>Array(r).fill(0));
    for(let i=0;i<r;i++) for(let j=0;j<c;j++) T[j][i]=A[i][j]; return T;}
  function matmul(A,B){ const r=A.length,c=B[0].length,m=B.length; const R=Array.from({length:r},()=>Array(c).fill(0));
    for(let i=0;i<r;i++) for(let j=0;j<c;j++){ let s=0; for(let k=0;k<m;k++) s+=A[i][k]*B[k][j]; R[i][j]=s; } return R;}
  function matvec(A,b){ const r=A.length,m=A[0].length; const v=new Array(r).fill(0);
    for(let i=0;i<r;i++){ let s=0; for(let k=0;k<m;k++) s+=A[i][k]*b[k]; v[i]=s; } return v;}
  function solveGauss(M,b){
    const n=M.length; const A=M.map(row=>row.slice()); const x=b.slice();
    for(let i=0;i<n;i++) A[i].push(x[i]);
    for(let i=0;i<n;i++){
      let piv=i; for(let r=i;r<n;r++) if(Math.abs(A[r][i])>Math.abs(A[piv][i])) piv=r;
      if(piv!==i) [A[i],A[piv]]=[A[piv],A[i]];
      const v=A[i][i]; if(Math.abs(v)<1e-12) throw new Error("Singular polyfit");
      for(let j=i;j<=n;j++) A[i][j]/=v;
      for(let r=0;r<n;r++){ if(r===i) continue; const f=A[r][i];
        for(let j=i;j<=n;j++) A[r][j]-=f*A[i][j];
      }
    }
    return A.map(row=>row[n]);
  }
}
function polyval(beta,x){ return x.map(v => beta.reduce((acc,c,p)=>acc + c*Math.pow(v,p),0)); }
function baselineIterative(x,yHat,order=2,maxIter=25,tolViolations=20){
  let beta=polyfitLS(x,yHat,order), base=polyval(beta,x), iter=0;
  while(iter<maxIter){
    for(let i=0;i<x.length;i++) if(base[i]>yHat[i]) base[i]=yHat[i];
    beta=polyfitLS(x,base,order); base=polyval(beta,x);
    const violations=base.reduce((acc,v,i)=>acc+(v>yHat[i]?1:0),0);
    if(violations<=tolViolations) break;
    iter++;
  }
  return base;
}

/* ====== Raman: helpers Lorentz ====== */
function lorentz(x,amp,cen,wid){ return x.map(v=> amp*wid*wid/((v-cen)*(v-cen)+wid*wid)); }
function sumLorentz(x,pars){ const y=new Array(x.length).fill(0);
  for(let p=0;p<pars.length;p+=3){ const amp=pars[p], cen=pars[p+1], wid=pars[p+2];
    for(let i=0;i<x.length;i++) y[i]+=amp*wid*wid/((x[i]-cen)*(x[i]-cen)+wid*wid);
  } return y; }
function fitLorentz(x,y, initPars, iters=1500, step={amp:0.15, cen:2.5, wid:3}, bounds){
  let pars=initPars.slice(), best=pars.slice(), bestLoss=mse(y,sumLorentz(x,best));
  for(let t=0;t<iters;t++){
    const cand=pars.slice();
    for(let p=0;p<cand.length;p+=3){
      cand[p]   += (Math.random()*2-1)*step.amp*Math.abs(cand[p]||1);
      cand[p+1] += (Math.random()*2-1)*step.cen;
      cand[p+2] += (Math.random()*2-1)*step.wid;
      if(bounds){
        cand[p]   = Math.max(bounds.ampMin, Math.min(bounds.ampMax, cand[p]));
        cand[p+1] = Math.max(bounds.cenMin, Math.min(bounds.cenMax, cand[p+1]));
        cand[p+2] = Math.max(bounds.widMin, Math.min(bounds.widMax, cand[p+2]));
      }
    }
    const loss=mse(y,sumLorentz(x,cand));
    if(loss<bestLoss || Math.random()<0.03){ pars=cand; if(loss<bestLoss){ best=cand.slice(); bestLoss=loss; } }
  }
  return {pars:best, loss:bestLoss};
}

/* ====== Raman: inicialización dinámica N-Lorentz ====== */
function nearestValue(region, cen){
  let best=0, bd=Infinity;
  for(let i=0;i<region.x.length;i++){
    const d=Math.abs(region.x[i]-cen); if(d<bd){ bd=d; best=i; }
  }
  return region.y[best];
}
function initLorentzPars(regionName, regionData, num){
  const centersDG = [1350, 1500, 1570, 1610, 1700];
  const centers2D = [2700, 2970, 3250, 3510, 3650];
  const centers = regionName === 'DG' ? centersDG : centers2D;
  const pars = [];
  for(let i=0;i<num;i++){
    const cen = centers[i];
    const amp = Math.max(1, (nearestValue(regionData, cen) - 50));
    const wid = regionName === 'DG' ? (i===1?20:80) : (i===1?20:80);
    pars.push(amp, cen, wid);
  }
  return pars;
}

/* ====== Raman (flujo principal) ====== */
async function parseJascoTxtRaman(file){
  const text = await file.text();
  const lines = text.split(/\r?\n/);
  const startIdx = lines.findIndex(l => l.trim().toUpperCase()==="XYDATA");
  if (startIdx<0) throw new Error("No se encontró XYDATA");
  const X=[],Y=[];
  for (let i=startIdx+1;i<lines.length;i++){
    const L=lines[i].trim(); if(!L || L.startsWith("#####")) break;
    const Ldot=L.replace(/,/g,'.');
    const parts=Ldot.split(/\s+/).filter(Boolean);
    if(parts.length>=2){
      const x=parseFloat(parts[0]), y=parseFloat(parts[1]);
      if(isFinite(x)&&isFinite(y)){ X.push(Math.round(x)); Y.push(y); }
    }
  }
  return {X,Y};
}
function sliceRegionByRange(X,Y,x0,x1){
  const i0 = nearestIndex(X,x0), i1 = nearestIndex(X,x1);
  if (i0===-1 || i1===-1 || i1<=i0) throw new Error("Rango fuera de X");
  return { x: X.slice(i0,i1+1), y: Y.slice(i0,i1+1) };
  function nearestIndex(arr,val){ let best=-1, bd=Infinity;
    for(let i=0;i<arr.length;i++){ const d=Math.abs(arr[i]-val); if(d<bd){ bd=d; best=i; } } return best; }
}
function qualityChecksDG(mediaXG, mediaIDIG, stXG, stIDIG){
  return [
    {k:'media XG rango', ok:(mediaXG<1580 && mediaXG>1574)},
    {k:'media ID/IG', ok:(mediaIDIG<1.5 && mediaIDIG>1.1)},
    {k:'desv est XG', ok:(stXG<2.5)},
    {k:'desv est IDIG', ok:(stIDIG<0.12)},
  ];
}
function qualityChecks2D(mediaX2D, mediaI2DIDG, st2D, stI2DIDG){
  return [
    {k:'media X2D rango', ok:(mediaX2D<2700 && mediaX2D>2690)},
    {k:'media I2D/(D+G’)', ok:(mediaI2DIDG<1.1 && mediaI2DIDG>0.8)},
    {k:'desv est X2D', ok:(st2D<2.5)},
    {k:'desv est I2D/(D+G’)', ok:(stI2DIDG<0.12)},
  ];
}

let lastReportRaman = '';
document.getElementById('runBtnRaman').addEventListener('click', async ()=>{
  try {
    const file = document.getElementById('fileInputRaman').files[0];
    if(!file){ alert("Carga un archivo Raman .txt"); return; }
    const {X,Y} = await parseJascoTxtRaman(file);
    document.getElementById('fileMetaRaman').textContent = `Puntos: ${X.length} | X[min,max]=[${Math.min(...X)}, ${Math.max(...X)}]`;

    const sgW=parseInt(document.getElementById('sgWindow').value,10);
    const sgO=parseInt(document.getElementById('sgOrder').value,10);
    const Yhat = savgol(Y, sgW, sgO);

    const bOrder=parseInt(document.getElementById('baselineOrder').value,10);
    const bIter =parseInt(document.getElementById('baselineIter').value,10);
    const base = baselineIterative(X, Yhat, bOrder, bIter, 20);
    const Ysub = Yhat.map((v,i)=> v - base[i]);
    const FL = trapz(X,Yhat)-trapz(X,Ysub);

    plotLines(document.getElementById('plotRaw'), [
      {x:X,y:Y,color:'#5e81ac'}, {x:X,y:Yhat,color:'#88c0d0'}
    ]);
    plotLines(document.getElementById('plotBaseline'), [
      {x:X,y:Yhat,color:'#8be9fd'}, {x:X,y:base,color:'#50fa7b'}, {x:X,y:Ysub,color:'#ff79c6'}
    ]);

    const dgStart=parseInt(document.getElementById('dgStart').value,10);
    const dgEnd  =parseInt(document.getElementById('dgEnd').value,10);
    const d2Start=parseInt(document.getElementById('d2Start').value,10);
    const d2End  =parseInt(document.getElementById('d2End').value,10);

    const DG = sliceRegionByRange(X,Ysub,dgStart,dgEnd);
    const D2 = sliceRegionByRange(X,Ysub,d2Start,d2End);

    const numDG = parseInt(document.getElementById('numLorentzDG').value,10);
    const num2D = parseInt(document.getElementById('numLorentz2D').value,10);

    let parsDG = initLorentzPars('DG', DG, numDG);
    let pars2D = initLorentzPars('2D', D2, num2D);

    let fitResDG = {pars:parsDG.slice(), loss:mse(DG.y,sumLorentz(DG.x,parsDG))};
    let fitRes2D = {pars:pars2D.slice(), loss:mse(D2.y,sumLorentz(D2.x,pars2D))};

    // Ajuste con límites y pasos diferenciados por región
    fitResDG = fitLorentz(DG.x, DG.y, parsDG, 1500, {amp:0.15, cen:2.5, wid:3},
      {ampMin:0, ampMax:1e6, cenMin:dgStart, cenMax:dgEnd, widMin:5, widMax:300});
    fitRes2D = fitLorentz(D2.x, D2.y, pars2D, 1500, {amp:0.15, cen:3.0, wid:3},
      {ampMin:0, ampMax:1e6, cenMin:d2Start, cenMax:d2End, widMin:5, widMax:300});

    const peaksFromPars=(x,pars)=>{ const arr=[]; for(let p=0;p<pars.length;p+=3) arr.push(lorentz(x,pars[p],pars[p+1],pars[p+2])); return arr; };
    const peaksDG = peaksFromPars(DG.x, fitResDG.pars);
    const peaks2D = peaksFromPars(D2.x, fitRes2D.pars);
    const sumDG = sumLorentz(DG.x, fitResDG.pars);
    const sum2D = sumLorentz(D2.x, fitRes2D.pars);
    const residDG = DG.y.map((v,i)=> v - sumDG[i]);
    const resid2D = D2.y.map((v,i)=> v - sum2D[i]);

    plotLines(document.getElementById('plotDG'), [
      {x:DG.x,y:DG.y,color:'#a3be8c'}, {x:DG.x,y:sumDG,color:'#e66'},
      ...peaksDG.map((py,k)=>({x:DG.x,y:py,color:['#8be9fd','#bd93f9','#50fa7b','#ff79c6','#f1fa8c'][k%5]})),
      {x:DG.x,y:residDG,color:'#999',width:1}
    ]);
    plotLines(document.getElementById('plot2D'), [
      {x:D2.x,y:D2.y,color:'#a3be8c'}, {x:D2.x,y:sum2D,color:'#e66'},
      ...peaks2D.map((py,k)=>({x:D2.x,y:py,color:['#8be9fd','#bd93f9','#50fa7b','#ff79c6','#f1fa8c'][k%5]})),
      {x:D2.x,y:resid2D,color:'#999',width:1}
    ]);

    // Métricas principales (se asume G y D presentes si N>=2)
    const getPeak = (pars, idx) => ({amp:pars[3*idx], cen:pars[3*idx+1], wid:pars[3*idx+2]});
    // Para DG: asumimos orden [D?, G?] - ubicamos G cerca de ~1570 por proximidad
    function findClosestCenter(pars, target){
      let bestIdx=0, bd=Infinity;
      for(let i=0;i<pars.length;i+=3){
        const cen=pars[i+1]; const d=Math.abs(cen-target);
        if(d<bd){ bd=d; bestIdx=i/3; }
      }
      return bestIdx;
    }
    const gIdx = findClosestCenter(fitResDG.pars, 1570);
    const dIdx = findClosestCenter(fitResDG.pars, 1350);

    const G = getPeak(fitResDG.pars, gIdx);
    const D = getPeak(fitResDG.pars, dIdx);
    const IDIG = (G.amp>0)? D.amp/G.amp : NaN;

    // Para 2D tomamos el pico más cercano a ~2700 cm-1
    const twoDIdx = findClosestCenter(fitRes2D.pars, 2700);
    const D2p = getPeak(fitRes2D.pars, twoDIdx);

    // Tomamos (D+G’) como el segundo pico más intenso de 2D (heurística simple)
    const amps2D = []; for(let i=0;i<fitRes2D.pars.length;i+=3) amps2D.push({i:i/3, amp:fitRes2D.pars[i]});
    amps2D.sort((a,b)=>b.amp-a.amp);
    const dgPrimeIdx = amps2D[1]?.i ?? twoDIdx;
    const DGprime = getPeak(fitRes2D.pars, dgPrimeIdx);
    const I2D_over_DGp = (DGprime.amp>0)? D2p.amp/DGprime.amp : NaN;

    const metricsDiv = document.getElementById('metricsRaman');
    metricsDiv.innerHTML='';
    [
      {k:'Área fluorescencia FL', v: FL.toFixed(2)},
      {k:'G (cen) [cm-1]', v: G.cen.toFixed(1)},
      {k:'D (cen) [cm-1]', v: D.cen.toFixed(1)},
      {k:'ID/IG', v: isFinite(IDIG)? IDIG.toFixed(3): 'NA'},
      {k:'2D (cen) [cm-1]', v: D2p.cen.toFixed(1)},
      {k:'I2D/(D+G’)', v: isFinite(I2D_over_DGp)? I2D_over_DGp.toFixed(3): 'NA'},
      {k:'DG MSE ajuste', v: fitResDG.loss.toExponential(3)},
      {k:'2D MSE ajuste', v: fitRes2D.loss.toExponential(3)},
      {k:'Lorentzianas DG', v: (fitResDG.pars.length/3).toString()},
      {k:'Lorentzianas 2D', v: (fitRes2D.pars.length/3).toString()},
    ].forEach(row=>{
      const d1=document.createElement('div'); d1.textContent=row.k;
      const d2=document.createElement('div'); d2.textContent=row.v;
      metricsDiv.appendChild(d1); metricsDiv.appendChild(d2);
    });

    const qDG = qualityChecksDG(G.cen, IDIG, 0, 0);
    const q2D = qualityChecks2D(D2p.cen, I2D_over_DGp, 0, 0);
    const qualityDiv = document.getElementById('qualityRaman'); qualityDiv.innerHTML='';
    function addQ(label, ok){
      const d1=document.createElement('div'); d1.textContent=label;
      const d2=document.createElement('div'); d2.textContent= ok?'cumple':'no cumple';
      d2.className = ok?'status-ok':'status-bad';
      qualityDiv.appendChild(d1); qualityDiv.appendChild(d2);
    }
    qDG.forEach(q=>addQ(`DG: ${q.k}`, q.ok));
    q2D.forEach(q=>addQ(`2D: ${q.k}`, q.ok));
    const allOk = [...qDG, ...q2D].every(q=>q.ok);
    lastReportRaman  = `Raman
Fluorescencia FL: ${FL.toFixed(2)}
G(cen): ${G.cen.toFixed(2)} cm-1
D(cen): ${D.cen.toFixed(2)} cm-1
ID/IG: ${(isFinite(IDIG)?IDIG.toFixed(4):'NA')}
2D(cen): ${D2p.cen.toFixed(2)} cm-1
I2D/(D+G’): ${(isFinite(I2D_over_DGp)?I2D_over_DGp.toFixed(4):'NA')}
Lorentzianas DG: ${(fitResDG.pars.length/3)}
Lorentzianas 2D: ${(fitRes2D.pars.length/3)}
Veredicto: ${allOk?'Cumple':'No cumple completamente'}
`;
  } catch(e){ alert(`Error Raman: ${e.message}`); }
});
document.getElementById('exportBtnRaman').addEventListener('click', ()=>{
  if (!lastReportRaman){ alert("Procesa Raman primero"); return; }
  const blob = new Blob([lastReportRaman], {type:'text/plain;charset=utf-8'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a'); a.href=url; a.download='Informe_Raman.txt'; a.click();
  URL.revokeObjectURL(url);
});

/* ====== XPS ====== */
document.getElementById('runBtnXPS').addEventListener('click', ()=>{
  const C = parseFloat(document.getElementById('xpsC').value||'0');
  const O = parseFloat(document.getElementById('xpsO').value||'0');
  const N = parseFloat(document.getElementById('xpsN').value||'0');
  const S = parseFloat(document.getElementById('xpsS').value||'0');
  const Otros = parseFloat(document.getElementById('xpsOtros').value||'0');
  const total = C+O+N+S+Otros;
  const norm = total>0 ? {C:C/total*100,O:O/total*100,N:N/total*100,S:S/total*100,Otros:Otros/total*100} : {C:0,O:0,N:0,S:0,Otros:0};
  document.getElementById('xpsMeta').textContent = `Total ingresado: ${total.toFixed(2)}% | Normalizado a 100%`;
  const table = document.getElementById('xpsTable'); table.innerHTML='';
  Object.entries(norm).forEach(([k,v])=>{
    const d1=document.createElement('div'); d1.textContent=`${k}`;
    const d2=document.createElement('div'); d2.textContent=`${v.toFixed(2)} % atómico`;
    table.appendChild(d1); table.appendChild(d2);
  });
  const canvas = document.getElementById('plotXPS'); const {ctx,dpr,W,H} = setupCanvas(canvas);
  ctx.clearRect(0,0,W,H);
  const labels=['C','O','N','S','Otros'], values=[norm.C,norm.O,norm.N,norm.S,norm.Otros];
  const padL=40,padB=30; const maxV = Math.max(100, ...values);
  const barW = (W/dpr - padL - 20) / (labels.length*1.5);
  for(let i=0;i<labels.length;i++){
    const x = padL + i*barW*1.5 + 10;
    const h = (H/dpr - padB - 10) * (values[i]/maxV);
    ctx.fillStyle='#42c8f5'; ctx.fillRect(x, H/dpr-padB-h, barW, h);
    ctx.fillStyle='#9aa6b2'; ctx.fillText(labels[i], x, H/dpr-padB+14);
    ctx.fillText(values[i].toFixed(1)+'%', x, H/dpr-padB-h-4);
  }
  const concl = (norm.C>=60 && norm.O<=20) ? 'Alta fracción de C y moderado O: consistente con grafito/GO reducido.'
                                           : 'Composición fuera de rango típico GO/G, analizar condiciones de síntesis.';
  document.getElementById('xpsConclusion').textContent = concl;
  window._XPS_state = {norm, concl};
});

/* ====== Utilidades de álgebra ====== */
function transpose(A){ const r=A.length,c=A[0].length; const T=Array.from({length:c},()=>Array(r).fill(0));
  for(let i=0;i<r;i++) for(let j=0;j<c;j++) T[j][i]=A[i][j]; return T;}
function matmul(A,B){ const r=A.length,c=B[0].length,m=B.length; const R=Array.from({length:r},()=>Array(c).fill(0));
  for(let i=0;i<r;i++) for(let j=0;j<c;j++){ let s=0; for(let k=0;k<m;k++) s+=A[i][k]*B[k][j]; R[i][j]=s; } return R;}
function matvec(A,b){ const r=A.length,m=A[0].length; const v=new Array(r).fill(0);
  for(let i=0;i<r;i++){ let s=0; for(let k=0;k<m;k++) s+=A[i][k]*b[k]; v[i]=s; } return v;}
function solveGauss(M,b){
  const n=M.length; const A=M.map(row=>row.slice()); const x=b.slice();
  for(let i=0;i<n;i++) A[i].push(x[i]);
  for(let i=0;i<n;i++){
    let piv=i; for(let r=i;r<n;r++) if(Math.abs(A[r][i])>Math.abs(A[piv][i])) piv=r;
    if(piv!==i) [A[i],A[piv]]=[A[piv],A[i]];
    const v=A[i][i]; if(Math.abs(v)<1e-12) throw new Error("Singular");
    for(let j=i;j<=n;j++) A[i][j]/=v;
    for(let r=0;r<n;r++){ if(r===i) continue; const f=A[r][i];
      for(let j=i;j<=n;j++) A[r][j]-=f*A[i][j];
    }
  }
  return A.map(row=>row[n]);
}

/* ====== Generar informe imprimible como PDF ====== */
document.getElementById('printBtn').addEventListener('click', ()=>{
  const gen = generalData || {};
  const genDiv = document.getElementById('printGeneral');
  genDiv.innerHTML = `
    <h2>Datos generales</h2>
    <pre class="mono">
Informe: ${gen.informe||'-'}
Fecha: ${gen.fecha||'-'}
Celda: ${gen.celda||'-'}
Reactor: ${gen.reactor||'-'}
Barra de grafito: ${gen.barra||'-'}
Voltaje máx [V]: ${gen.vmax||'-'}
Corriente máx [A]: ${gen.imax||'-'}
Temperatura máx [°C]: ${gen.tmax||'-'}
Tiempo de exfoliación [min]: ${gen.tiempo||'-'}
    </pre>
  `;
  function canvasToImg(id,title){
    const c = document.getElementById(id); if (!c) return '';
    const data = c.toDataURL('image/png');
    return `<h3>${title}</h3><img class="print-img" src="${data}" />`;
  }
  const metR = document.getElementById('metricsRaman').innerText || '';
  const verR = document.getElementById('qualityRaman').innerText || '';
  document.getElementById('printRaman').innerHTML = `
    ${canvasToImg('plotRaw','Crudo vs suavizado')}
    ${canvasToImg('plotBaseline','Baseline y sustracción')}
    ${canvasToImg('plotDG','Región DG: ajuste y residuos')}
    ${canvasToImg('plot2D','Región 2D: ajuste y residuos')}
    <h3>Métricas y veredictos</h3>
    <pre class="mono">${metR}\n\n${verR}</pre>
  `;
   const cXPS = document.getElementById('plotXPS');
  const tblX = document.getElementById('xpsTable').innerText || '';
  const conclX = document.getElementById('xpsConclusion').innerText || '';
  const dataXPS = cXPS ? cXPS.toDataURL('image/png') : '';
  document.getElementById('printXPS').innerHTML = `
    <h3>Gráfico XPS</h3>
    ${dataXPS? `<img class="print-img" src="${dataXPS}" />` : '<div>No procesado</div>'}
    <h3>Tabla</h3>
    <pre class="mono">${tblX}</pre>
    <h3>Conclusión XPS</h3>
    <pre class="mono">${conclX}</pre>
  `;
  const verRtxt = document.getElementById('qualityRaman').innerText || '';
    const xpsCon  = conclX || '';
  const finalConcl = `
Raman: ${verRtxt.includes('cumple') ? 'Cumple criterios principales.' : 'Resultados parciales.'}
XPS: ${xpsCon}
  `.trim();
  document.getElementById('printConclusion').innerHTML = `<pre class="mono">${finalConcl}</pre>`;
  window.print();
});