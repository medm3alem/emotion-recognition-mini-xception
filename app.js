import * as tf from '@tensorflow/tfjs';
import * as blazeface from 'https://esm.sh/@tensorflow-models/blazeface@0.1.0?external=@tensorflow/tfjs';

const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const ctx = overlay.getContext('2d');
const startButton = document.getElementById('start');
const statusText = document.getElementById('status');
const emotionText = document.getElementById('emotion');
const probabilitiesContainer = document.getElementById('probabilities');
const placeholder = document.getElementById('placeholder');
const faceLatency = document.getElementById('faceLatency');
const modelLatency = document.getElementById('modelLatency');
const totalLatency = document.getElementById('totalLatency');
const fpsText = document.getElementById('fps');
const LABELS = ['Colère','Dégoût','Peur','Joie','Tristesse','Surprise','Neutre'];
let emotionModel, faceModel, stream, running = false;
const rows = [];
function initializeProbabilities() {
  LABELS.forEach(label => {
    const row = document.createElement('div'); row.className = 'probability-row';
    const name = document.createElement('span'); name.className='name'; name.textContent=label;
    const track=document.createElement('div'); track.className='bar-track';
    const fill=document.createElement('div'); fill.className='bar-fill'; track.appendChild(fill);
    const value=document.createElement('span'); value.className='value'; value.textContent='0.0 %';
    row.append(name,track,value); probabilitiesContainer.appendChild(row);
    rows.push({row,fill,value});
  });
}
function updateProbabilities(probabilities) {
  const values=Array.from(probabilities);
  const bestIndex=values.indexOf(Math.max(...values));
  values.forEach((p,i)=>{const pct=p*100; rows[i].fill.style.width=`${pct}%`; rows[i].value.textContent=`${pct.toFixed(1)} %`; rows[i].row.classList.toggle('active',i===bestIndex);});
  emotionText.textContent=`${LABELS[bestIndex]} — ${(values[bestIndex]*100).toFixed(1)} %`;
}
async function loadModels() {
  try {
    statusText.textContent='Initialisation de TensorFlow.js...';
    await tf.ready(); console.log('Backend :',tf.getBackend());
    statusText.textContent='Chargement de Mini-Xception...';
    emotionModel=await tf.loadGraphModel('./model/model.json');
    console.log('Mini-Xception :',emotionModel.inputs,emotionModel.outputs);
    statusText.textContent='Chargement du détecteur de visages...';
    faceModel=await blazeface.load();
    statusText.textContent='Modèles prêts. Activez votre webcam.';
    startButton.textContent='Activer la webcam'; startButton.disabled=false;
  } catch(error) {
    console.error('Erreur de chargement :',error);
    statusText.textContent=`Erreur de chargement : ${error.message}`;
    startButton.textContent='Erreur de chargement';
  }
}
async function startCamera() {
  try {
    startButton.disabled=true; statusText.textContent="Demande d'accès à la webcam...";
    stream=await navigator.mediaDevices.getUserMedia({video:{width:{ideal:640},height:{ideal:480},facingMode:'user'},audio:false});
    video.srcObject=stream; await video.play();
    overlay.width=video.videoWidth; overlay.height=video.videoHeight;
    placeholder.classList.add('hidden'); running=true;
    startButton.textContent='Webcam active'; statusText.textContent='Analyse en temps réel...';
    void detectLoop();
  } catch(error) {
    console.error('Erreur webcam :',error); statusText.textContent=`Impossible d'accéder à la webcam : ${error.message}`; startButton.disabled=false;
  }
}
async function predictEmotion(x,y,width,height) {
  const input=tf.tidy(()=>tf.browser.fromPixels(video,1).slice([y,x,0],[height,width,1]).resizeBilinear([48,48]).toFloat().div(255).expandDims(0));
  let output;
  try {
    output=await emotionModel.executeAsync(input);
    const prediction=Array.isArray(output)?output[0]:output;
    updateProbabilities(await prediction.data());
  } finally {
    input.dispose();
    if(output) (Array.isArray(output)?output:[output]).forEach(t=>t.dispose());
  }
}
async function detectLoop() {
  while(running) {
    try {
      const tStart = performance.now();
      const faces=await faceModel.estimateFaces(video,false);
      const tFace = performance.now();
      faceLatency.textContent = `${(tFace - tStart).toFixed(1)} ms`;
      let classificationPerformed = false;
      ctx.clearRect(0,0,overlay.width,overlay.height);
      if(!faces.length) emotionText.textContent='Aucun visage détecté';
      else {
        const face=faces[0]; const [x1,y1]=face.topLeft; const [x2,y2]=face.bottomRight;
        const x=Math.max(0,Math.floor(x1)),y=Math.max(0,Math.floor(y1));
        const right=Math.min(video.videoWidth,Math.ceil(x2)),bottom=Math.min(video.videoHeight,Math.ceil(y2));
        const width=right-x,height=bottom-y;
        if(width>0&&height>0) {
          ctx.strokeStyle='#22c55e';ctx.lineWidth=3;ctx.strokeRect(x,y,width,height);
          await predictEmotion(x,y,width,height);
          classificationPerformed = true;
        }
      }
      const tEnd = performance.now();
      modelLatency.textContent = classificationPerformed ? `${(tEnd - tFace).toFixed(1)} ms` : '-- ms';
      totalLatency.textContent = `${(tEnd - tStart).toFixed(1)} ms`;
      fpsText.textContent = `${(1000 / Math.max(tEnd - tStart, 1)).toFixed(1)} FPS`;
    } catch(error) {
      console.error('Erreur d’analyse :',error);statusText.textContent=`Erreur de prédiction : ${error.message}`;running=false;break;
    }
    await new Promise(resolve=>setTimeout(resolve,150));
  }
}
window.addEventListener('beforeunload',()=>{running=false;if(stream)stream.getTracks().forEach(track=>track.stop());});
startButton.addEventListener('click',startCamera);
initializeProbabilities();void loadModels();
