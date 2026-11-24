import sqlite3

# Connect to SQLite (this will create a file named 'wardrobe.db' in your folder)
conn = sqlite3.connect('wardrobe.db')
cursor = conn.cursor()

# Create the Clothing_Items table
cursor.execute('''
CREATE TABLE IF NOT EXISTS Clothing_Items (
    item_id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_path TEXT NOT NULL,
    category TEXT NOT NULL,
    material TEXT,
    color TEXT,
    weather_suitability TEXT,
    is_waterproof BOOLEAN,
    warmth_level INTEGER,
    embedding_vector BLOB 
)
''')

conn.commit()
conn.close()

print("Database 'wardrobe.db' and Clothing_Items table created successfully!")