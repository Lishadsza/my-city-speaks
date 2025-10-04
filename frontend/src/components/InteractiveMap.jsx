import React, { useState, useEffect } from 'react';
import { MapContainer, TileLayer, Marker } from 'react-leaflet';
import { Loader, AlertTriangle, MapPin, XCircle } from 'lucide-react';

const InteractiveMap = ({ onCitySelect, CityPanelComponent }) => {
  const [cities, setCities] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedCity, setSelectedCity] = useState(null);

  const INDIA_CENTER = [22.3511, 78.6677]; 
  const FLASK_API_URL = 'http://127.0.0.1:5000'; 

  useEffect(() => {
    setIsLoading(true);
    setError(null);
    fetch(`${FLASK_API_URL}/api/cities`) 
      .then(res => {
        if (!res.ok) {
            throw new Error(`HTTP error! status: ${res.status}`);
        }
        return res.json();
      })
      .then(data => {
        const validCities = data.filter(city => city && city.lat !== null && city.lng !== null);

        if (validCities.length === 0) {
            setError("Successfully connected, but no valid city coordinates found in the database. Please check Lat/Lng values.");
        }
        setCities(validCities);
      })
      .catch(err => {
        console.error('Error fetching cities:', err);
        setError("Failed to load cities from backend. Check if Flask server is running on port 5000.");
      })
      .finally(() => {
        setIsLoading(false);
      });
  }, []);

  const handleMarkerClick = (city) => {
    setSelectedCity(city);
    onCitySelect(city);
  };

  const handleUploadSuccess = () => {
    console.log("Upload completed, refreshing city panel data.");
  };

  //  Loading State
  if (isLoading) {
    return (
      <div className="relative flex flex-col items-center justify-center h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 overflow-hidden">
        <div className="absolute inset-0">
          <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-blue-500/10 rounded-full blur-3xl animate-pulse"></div>
          <div className="absolute bottom-1/4 right-1/4 w-80 h-80 bg-purple-500/10 rounded-full blur-3xl animate-pulse" style={{animationDelay: '1s'}}></div>
        </div>
        
        <div className="relative z-10 flex flex-col items-center">
          <div className="relative mb-6">
            <div className="absolute inset-0 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full blur-xl opacity-50 animate-pulse"></div>
            <div className="relative w-20 h-20 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center">
              <Loader className="w-10 h-10 text-white animate-spin" />
            </div>
          </div>
          <p className="text-2xl font-bold text-white mb-2">Loading Map Data</p>
          <p className="text-gray-400">Fetching city locations...</p>
          
          <div className="flex gap-2 mt-4">
            <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce"></div>
            <div className="w-2 h-2 bg-purple-400 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
            <div className="w-2 h-2 bg-pink-400 rounded-full animate-bounce" style={{animationDelay: '0.4s'}}></div>
          </div>
        </div>
      </div>
    );
  }

  //  Error State
  if (error) {
    return (
      <div className="relative flex flex-col items-center justify-center h-screen bg-gradient-to-br from-slate-900 via-red-900/30 to-slate-900 overflow-hidden">
        <div className="absolute inset-0">
          <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-red-500/10 rounded-full blur-3xl animate-pulse"></div>
        </div>
        
        <div className="relative z-10 flex flex-col items-center max-w-2xl px-8">
          <div className="relative mb-6">
            <div className="absolute inset-0 bg-red-500 rounded-full blur-xl opacity-30"></div>
            <div className="relative w-20 h-20 bg-gradient-to-br from-red-500 to-red-600 rounded-full flex items-center justify-center border-2 border-white/10">
              <AlertTriangle className="w-10 h-10 text-white" />
            </div>
          </div>
          
          <h3 className="text-2xl font-bold text-white mb-3">Connection Error</h3>
          <div className="bg-red-500/10 backdrop-blur-sm border border-red-500/30 rounded-2xl p-6 text-center">
            <p className="text-red-200 leading-relaxed">{error}</p>
          </div>
          
          <button 
            onClick={() => window.location.reload()} 
            className="mt-6 px-8 py-3 bg-red-500/20 hover:bg-red-500/30 border border-red-500/50 text-white rounded-xl font-semibold transition-all duration-300 hover:scale-105"
          >
            Retry Connection
          </button>
        </div>
      </div>
    );
  }

  // Full Screen Map
  return (
    <>
      <MapContainer 
        center={INDIA_CENTER} 
        zoom={5} 
        minZoom={4} 
        scrollWheelZoom={true}
        style={{ height: '100vh', width: '100vw', position: 'fixed', top: 0, left: 0, zIndex: 0 }}
        role="application"
      >
        <TileLayer
          attribution='&copy; <a href="http://osm.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
          maxZoom={18}
        />

        {cities.map(city => (
          <Marker 
            key={city.id} 
            position={[parseFloat(city.lat), parseFloat(city.lng)]}
            title={`Location: ${city.name.trim()}. Click to view audio recordings.`}
            eventHandlers={{
              click: () => handleMarkerClick(city), 
            }}
          />
        ))}
      </MapContainer>

      {/* centered city panel*/}
      {selectedCity && CityPanelComponent && (
        <div className="fixed inset-0 z-[1000] bg-black/60 backdrop-blur-sm overflow-y-auto">
          <div className="min-h-screen flex items-center justify-center p-4 py-8">
            <div className="bg-white rounded-3xl shadow-2xl max-w-5xl w-full my-8">
              {/* Header */}
              <div className="p-6 bg-gradient-to-r from-purple-50 to-blue-50 rounded-t-3xl flex justify-between items-start sticky top-0 z-10">
                <div className="flex items-center gap-4">
                  <div className="w-16 h-16 bg-gradient-to-br from-blue-500 to-purple-600 rounded-2xl flex items-center justify-center shadow-xl">
                    <MapPin className="w-8 h-8 text-white" />
                  </div>
                  <div>
                    <h2 className="text-3xl font-black text-gray-800 mb-1">{selectedCity.name}</h2>
                    <p className="text-purple-600 text-sm font-medium">Contribution Hub</p>
                  </div>
                </div>
                <button 
                  onClick={() => setSelectedCity(null)}
                  className="p-2 text-gray-400 hover:text-gray-800 hover:bg-gray-100 rounded-lg transition-all duration-300"
                  title="Close Panel"
                >
                  <XCircle className="w-7 h-7" />
                </button>
              </div>

              {/* Content - CityPanel */}
              <div className="p-8 bg-white rounded-b-3xl">
                <CityPanelComponent 
                  city={selectedCity} 
                  onUploadSuccess={handleUploadSuccess}
                />
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default InteractiveMap;