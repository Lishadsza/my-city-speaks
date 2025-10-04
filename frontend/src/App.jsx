import React, { useState } from 'react';
import { Home as HomeIcon, Upload, Phone, Globe,Home  } from 'lucide-react';
import UploadPage from './pages/UploadPage.jsx';
import Contact from './pages/contact.jsx'; 
import InteractiveMap from './components/InteractiveMap';
import CityPanel from './components/CityPanel';
import L from 'leaflet'; 
import { Routes, Route } from 'react-router-dom'; 
import Navbar from './components/Navbar'; 
import MapAndContributePage from './pages/MapAndContributePage';
import HomePage from './pages/HomePage.jsx';


// ensures the map markers display correctly in the browser
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
    iconRetinaUrl: 'https://unpkg.com/leaflet@1.7.1/dist/images/marker-icon-2x.png',
    iconUrl: 'https://unpkg.com/leaflet@1.7.1/dist/images/marker-icon.png',
    shadowUrl: 'https://unpkg.com/leaflet@1.7.1/dist/images/marker-shadow.png',
});

// Static list of languages for the dropdown
const LANGUAGES = ['Hindi', 'Marathi', 'Tamil', 'Telugu', 'Kannada', 'Bengali', 'Gujarati', 'Odia', 'Malayalam', 'Punjabi', 'Assamese', 'Other'];

// Component that handles the map/panel switching logic for the Home route
const MapInterface = ({ languages }) => {
    const [selectedCity, setSelectedCity] = useState(null); 
    // Function passed down to CityPanel to trigger re-fetching of recordings after an upload
    const handleUploadSuccess = () => {
     // CityPanel handles its own re-fetch logic when this is called
    };
    const handleCitySelect = (cityData) => {
        setSelectedCity(cityData);
    };

    const handleClosePanel = () => {
        setSelectedCity(null);
    };

    return (
        <div className="pt-24 min-h-screen container mx-auto p-4 w-full max-w-7xl">
            {selectedCity ? (
                <div className="flex justify-center">
                    <div className="w-full max-w-3xl bg-white p-6 shadow-xl rounded-xl">
                        <button 
                            onClick={handleClosePanel} 
                            className="mb-4 text-sm font-semibold text-blue-600 hover:text-blue-800 transition duration-150 ease-in-out flex items-center"
                        >
                            &larr; Back to Map View
                        </button>
                        <CityPanel 
                            city={selectedCity} 
                            onUploadSuccess={handleUploadSuccess} 
                            languages={languages}
                        />
                    </div>
                </div>
            ) : (
                <div className="shadow-2xl rounded-xl overflow-hidden border border-gray-200">
                    <h2 className="text-xl font-bold text-gray-800 p-4 bg-gray-100 border-b">
                        Interactive City Map (Click a marker to explore)
                    </h2>
                    <InteractiveMap onCitySelect={handleCitySelect} />
                </div>
            )}
        </div>
    );
};
export default function App() {
    return (
        <div className="min-h-screen bg-gray-100"> 
                        <Navbar /> 

            {/* 2. Main Content Area */}
            <main>
                <Routes>
                    {/*  Home Route */}
                    <Route path="/" element={<HomePage />} />
                    
                    {/* Map Route (Uses the combined container) */}
                    <Route path="/map" element={<MapAndContributePage />} /> 
                    
                    {/*  Analysis Route (Uses  UploadPage component) */}
                    <Route path="/analysis" element={<UploadPage />} />
                    
                    {/*Contact Me Route */}
                    <Route path="/contact" element={<Contact />} />
                </Routes>
            </main>
        </div>
    );
}