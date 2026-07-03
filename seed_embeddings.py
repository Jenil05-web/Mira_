import os
import pickle
from dotenv import load_dotenv
from openai import OpenAI
from sqlalchemy import create_engine, text

load_dotenv()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

db_url = "postgresql://postgres.tfrlbotgzxxdqwviemiz:EtnV1l2E7359NakT@aws-1-ap-southeast-2.pooler.supabase.com:6543/postgres"
engine = create_engine(db_url)

with open("./mira_data/faiss_metadata.pkl", "rb") as f:
    guidelines = pickle.load(f)

def embed(text):
    r = client.embeddings.create(model="text-embedding-3-small", input=[text])
    return r.data[0].embedding

with engine.connect() as conn:
    for i, g in enumerate(guidelines):
        vec = embed(g["text"])
        vec_str = "[" + ",".join(str(x) for x in vec) + "]"
        conn.execute(text("""
            INSERT INTO mira_embeddings (source, topic, content, embedding, hospital_id)
            VALUES (:source, :topic, :content, CAST(:vec AS vector), 'global')
        """), {
            "source": g["source"],
            "topic":  g["topic"],
            "content": g["text"],
            "vec":    vec_str
        })
        print(f"  [{i+1}/{len(guidelines)}] {g['topic']}")
    conn.commit()
    print(f"\n✅ Seeded {len(guidelines)} guideline chunks into Supabase pgvector")