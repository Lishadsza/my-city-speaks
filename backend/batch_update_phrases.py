#!/usr/bin/env python3
"""
Quick batch update script for phrases
Modify the PHRASE_UPDATES dictionary below with your changes
"""

import os
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


# Add the phrases you want to add/update here
NEW_PHRASES = [
    "Thank you",
    "You're welcome",
    "Excuse me",
    "I'm sorry",
    "How much does this cost?",
    "I don't understand",
    "What time is it?",
]

# update existing phrases by ID:
# Format: {phrase_id: "new_text"}
PHRASE_UPDATES = {
    # Example: 1: "Updated phrase text",
    # Example: 2: "Another updated phrase",
}
# ===== END MODIFICATION SECTION =====

def add_new_phrases():
    """Add all new phrases from NEW_PHRASES list"""
    print("Adding new phrases...")
    for phrase in NEW_PHRASES:
        try:
            # Check if phrase already exists
            existing = supabase.table("phrases").select("id").eq("english_text", phrase).execute()
            if existing.data:
                print(f"⚠️  Phrase already exists: '{phrase}'")
                continue
                
            # Add new phrase
            response = supabase.table("phrases").insert({"english_text": phrase}).execute()
            print(f"✅ Added: '{phrase}'")
        except Exception as e:
            print(f"❌ Error adding '{phrase}': {e}")

def update_existing_phrases():
    """Update existing phrases from PHRASE_UPDATES dictionary"""
    if not PHRASE_UPDATES:
        print("No phrase updates specified.")
        return
        
    print("Updating existing phrases...")
    for phrase_id, new_text in PHRASE_UPDATES.items():
        try:
            response = supabase.table("phrases").update({"english_text": new_text}).eq("id", phrase_id).execute()
            if response.data:
                print(f" Updated ID {phrase_id}: '{new_text}'")
            else:
                print(f" No phrase found with ID {phrase_id}")
        except Exception as e:
            print(f" Error updating ID {phrase_id}: {e}")

def list_current_phrases():
    """List all current phrases"""
    try:
        response = supabase.table("phrases").select("id, english_text").order("id").execute()
        print("\n=== Current Phrases in Database ===")
        for phrase in response.data:
            print(f"ID: {phrase['id']:2d} | '{phrase['english_text']}'")
        print(f"\nTotal: {len(response.data)} phrases")
    except Exception as e:
        print(f"Error fetching phrases: {e}")

def main():
    print("🔧 Batch Phrase Update Tool")
    print("=" * 40)
    
    # Show current phrases
    list_current_phrases()
    
    # Ask for confirmation
    print(f"\nReady to:")
    print(f"- Add {len(NEW_PHRASES)} new phrases")
    print(f"- Update {len(PHRASE_UPDATES)} existing phrases")
    
    confirm = input("\nProceed? (y/N): ").strip().lower()
    if confirm != 'y':
        print(" Operation cancelled")
        return
    
    # Execute updates
    add_new_phrases()
    update_existing_phrases()
    
    # Show final result
    print("\n" + "=" * 40)
    list_current_phrases()
    print("✅ Batch update completed!")

if __name__ == "__main__":
    main()