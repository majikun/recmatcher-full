# Recmatcher 本地 Web UI（打包骨架）
后端 FastAPI（uvicorn 127.0.0.1:8787），前端 Vite/React（http://127.0.0.1:5173）。

## 后端
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./run.sh

## 前端
cd frontend
npm i
npm run dev
