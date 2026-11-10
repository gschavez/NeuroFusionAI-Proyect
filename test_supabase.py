import os
from dotenv import load_dotenv
from supabase import create_client, Client

# 1. Load environment variables from .env
load_dotenv()

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")

if not url or not key:
    print("❌ Error: Missing Supabase credentials in .env file.")
    print("Make sure your .env file has SUPABASE_URL and SUPABASE_KEY.")
else:
    print("⏳ Connecting to Supabase...")
    try:
        # 2. Create the Supabase client
        supabase: Client = create_client(url, key)
        
        # 3. Test inserting data into the model_metrics table
        print("⏳ Inserting test data...")
        response = supabase.table("model_metrics").insert({
            "module_name": "Audio",
            "algorithm_name": "Support Vector Machine (SVM)",
            "accuracy": 0.7850,
            "f1_score": 0.7610,
            "is_best_model": True
        }).execute()
        
        # 4. Print success message and inserted data
        print("✅ Connection successful! Test data inserted.")
        print("Data returned by Supabase:")
        print(response.data)
        
    except Exception as e:
        print("❌ Error connecting to Supabase:")
        print(e)