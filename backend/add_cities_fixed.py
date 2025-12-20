#!/usr/bin/env python3
"""
Script to add Indian cities with only name, lat, lng (no state column)
Usage: python add_cities_fixed.py
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

# Major Indian Cities (without state column)
INDIAN_CITIES = [
    {"name": "Mumbai", "lat": 19.0760, "lng": 72.8777},
    {"name": "Delhi", "lat": 28.7041, "lng": 77.1025},
    {"name": "Bengaluru", "lat": 12.9716, "lng": 77.5946},
    {"name": "Kolkata", "lat": 22.5726, "lng": 88.3639},
    {"name": "Chennai", "lat": 13.0827, "lng": 80.2707},
    {"name": "Hyderabad", "lat": 17.3850, "lng": 78.4867},
    {"name": "Pune", "lat": 18.5204, "lng": 73.8567},
    {"name": "Ahmedabad", "lat": 23.0225, "lng": 72.5714},
    {"name": "Surat", "lat": 21.1702, "lng": 72.8311},
    {"name": "Jaipur", "lat": 26.9124, "lng": 75.7873},
    {"name": "Lucknow", "lat": 26.8467, "lng": 80.9462},
    {"name": "Kanpur", "lat": 26.4499, "lng": 80.3319},
    {"name": "Nagpur", "lat": 21.1458, "lng": 79.0882},
    {"name": "Indore", "lat": 22.7196, "lng": 75.8577},
    {"name": "Bhopal", "lat": 23.2599, "lng": 77.4126},
    {"name": "Visakhapatnam", "lat": 17.6868, "lng": 83.2185},
    {"name": "Patna", "lat": 25.5941, "lng": 85.1376},
    {"name": "Vadodara", "lat": 22.3072, "lng": 73.1812},
    {"name": "Coimbatore", "lat": 11.0168, "lng": 76.9558},
    {"name": "Agra", "lat": 27.1767, "lng": 78.0081},
    {"name": "Ludhiana", "lat": 30.9010, "lng": 75.8573},
    {"name": "Madurai", "lat": 9.9252, "lng": 78.1198},
    {"name": "Guwahati", "lat": 26.1445, "lng": 91.7362},
    {"name": "Chandigarh", "lat": 30.7333, "lng": 76.7794},
    {"name": "Thiruvananthapuram", "lat": 8.5241, "lng": 76.9366},
    {"name": "Solapur", "lat": 17.6599, "lng": 75.9064},
    {"name": "Hubballi", "lat": 15.3647, "lng": 75.1240},
    {"name": "Tiruchirappalli", "lat": 10.7905, "lng": 78.7047},
    {"name": "Bareilly", "lat": 28.3670, "lng": 79.4304},
    {"name": "Mysuru", "lat": 12.2958, "lng": 76.6394},
    {"name": "Tiruppur", "lat": 11.1085, "lng": 77.3411},
    {"name": "Gwalior", "lat": 26.2183, "lng": 78.1828},
    {"name": "Jabalpur", "lat": 23.1815, "lng": 79.9864},
    {"name": "Aligarh", "lat": 27.8974, "lng": 78.0880},
    {"name": "Guntur", "lat": 16.3067, "lng": 80.4365},
    {"name": "Bhiwandi", "lat": 19.3002, "lng": 73.0635},
    {"name": "Saharanpur", "lat": 29.9680, "lng": 77.5552},
    {"name": "Gorakhpur", "lat": 26.7606, "lng": 83.3732},
    {"name": "Bikaner", "lat": 28.0229, "lng": 73.3119},
    {"name": "Amravati", "lat": 20.9374, "lng": 77.7796},
    {"name": "Noida", "lat": 28.5355, "lng": 77.3910},
    {"name": "Jamshedpur", "lat": 22.8046, "lng": 86.2029},
    {"name": "Bhilai", "lat": 21.1938, "lng": 81.3509},
    {"name": "Cuttack", "lat": 20.4625, "lng": 85.8828},
    {"name": "Firozabad", "lat": 27.1592, "lng": 78.3957},
    {"name": "Kochi", "lat": 9.9312, "lng": 76.2673},
    {"name": "Nellore", "lat": 14.4426, "lng": 79.9865},
    {"name": "Bhavnagar", "lat": 21.7645, "lng": 72.1519},
    {"name": "Dehradun", "lat": 30.3165, "lng": 78.0322},
    {"name": "Durgapur", "lat": 23.4800, "lng": 87.3119},
    {"name": "Asansol", "lat": 23.6739, "lng": 86.9524},
    {"name": "Rourkela", "lat": 22.2604, "lng": 84.8536},
    {"name": "Nanded", "lat": 19.1383, "lng": 77.3210},
    {"name": "Kolhapur", "lat": 16.7050, "lng": 74.2433},
    {"name": "Ajmer", "lat": 26.4499, "lng": 74.6399},
    {"name": "Akola", "lat": 20.7002, "lng": 77.0082},
    {"name": "Gulbarga", "lat": 17.3297, "lng": 76.8343},
    {"name": "Jamnagar", "lat": 22.4707, "lng": 70.0577},
    {"name": "Ujjain", "lat": 23.1765, "lng": 75.7885},
    {"name": "Loni", "lat": 28.7333, "lng": 77.2833},
    {"name": "Siliguri", "lat": 26.7271, "lng": 88.3953},
    {"name": "Jhansi", "lat": 25.4484, "lng": 78.5685},
    {"name": "Ulhasnagar", "lat": 19.2215, "lng": 73.1645},
    {"name": "Jammu", "lat": 32.7266, "lng": 74.8570},
    {"name": "Sangli", "lat": 16.8524, "lng": 74.5815},
    {"name": "Mangaluru", "lat": 12.9141, "lng": 74.8560},
    {"name": "Erode", "lat": 11.3410, "lng": 77.7172},
    {"name": "Belgaum", "lat": 15.8497, "lng": 74.4977},
    {"name": "Ambattur", "lat": 13.1143, "lng": 80.1548},
    {"name": "Tirunelveli", "lat": 8.7139, "lng": 77.7567},
    {"name": "Malegaon", "lat": 20.5579, "lng": 74.5287},
    {"name": "Gaya", "lat": 24.7914, "lng": 85.0002},
    {"name": "Jalgaon", "lat": 21.0077, "lng": 75.5626},
    {"name": "Udaipur", "lat": 24.5854, "lng": 73.7125},
    {"name": "Maheshtala", "lat": 22.5049, "lng": 88.2482},
    {"name": "Srinagar", "lat": 34.0837, "lng": 74.7973},
    {"name": "Aurangabad", "lat": 19.8762, "lng": 75.3433},
    {"name": "Dhanbad", "lat": 23.7957, "lng": 86.4304},
    {"name": "Amritsar", "lat": 31.6340, "lng": 74.8723},
    {"name": "Allahabad", "lat": 25.4358, "lng": 81.8463},
    {"name": "Ranchi", "lat": 23.3441, "lng": 85.3096},
    {"name": "Howrah", "lat": 22.5958, "lng": 88.2636},
    {"name": "Jalandhar", "lat": 31.3260, "lng": 75.5762},
    {"name": "Jodhpur", "lat": 26.2389, "lng": 73.0243},
    {"name": "Raipur", "lat": 21.2514, "lng": 81.6296},
    {"name": "Kota", "lat": 25.2138, "lng": 75.8648},
    {"name": "Gurgaon", "lat": 28.4595, "lng": 77.0266},
    {"name": "Moradabad", "lat": 28.8386, "lng": 78.7733},
]

def add_cities():
    """Add cities to database (only name, lat, lng)"""
    print(f Adding {len(INDIAN_CITIES)} Indian cities...")
    
    added_count = 0
    skipped_count = 0
    error_count = 0
    
    for city in INDIAN_CITIES:
        try:
            # Check if city already exists
            existing = supabase.table("cities").select("id").eq("name", city['name']).execute()
            if existing.data:
                print(f"{city['name']} already exists")
                skipped_count += 1
                continue
            
            # Add city (only name, lat, lng)
            response = supabase.table("cities").insert(city).execute()
            print(f"✅ Added: {city['name']} ({city['lat']:.4f}, {city['lng']:.4f})")
            added_count += 1
            
        except Exception as e:
            print(f"Error adding {city['name']}: {e}")
            error_count += 1
    
    print(f"\n Summary:")
    print(f" Added: {added_count} cities")
    print(f" Skipped: {skipped_count} cities (already exist)")
    print(f"Errors: {error_count} cities")

def list_cities():
    """List all cities in database"""
    try:
        response = supabase.table("cities").select("*").order("name").execute()
        print(f"\n=== All Cities in Database ({len(response.data)}) ===")
        for city in response.data:
            print(f"ID: {city.get('id', 'N/A'):3d} | {city['name']:20s} | ({city['lat']:8.4f}, {city['lng']:8.4f})")
    except Exception as e:
        print(f"Error listing cities: {e}")

def main():
    print("🇮🇳 Indian Cities Geo-tagging (Fixed)")
    print("=" * 45)
    
    confirm = input(f"Add {len(INDIAN_CITIES)} Indian cities to database? (y/N): ").strip().lower()
    if confirm != 'y':
        print(" Cancelled")
        return
    
    add_cities()
    
    # Show final result
    print("\n" + "=" * 45)
    list_cities()
    print("✅ Done!")

if __name__ == "__main__":
    main()