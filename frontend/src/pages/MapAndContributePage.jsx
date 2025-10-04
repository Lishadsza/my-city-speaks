import React, { useState } from 'react';
import InteractiveMap from '../components/InteractiveMap';
import CityPanel from '../components/CityPanel';
import { MapPin, XCircle, ChevronLeft, MousePointerClick,  Network} from 'lucide-react';
import 'leaflet/dist/leaflet.css';
// Leaflet icon 
import L from 'leaflet';
import iconRetinaUrl from 'leaflet/dist/images/marker-icon-2x.png';
import iconUrl from 'leaflet/dist/images/marker-icon.png';
import shadowUrl from 'leaflet/dist/images/marker-shadow.png';
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
    iconRetinaUrl: iconRetinaUrl,
    iconUrl: iconUrl,
    shadowUrl: shadowUrl,
});

const MapAndContributePage = () => {
    const [selectedCity, setSelectedCity] = useState(null);

    const handleCitySelect = (city) => {
        setSelectedCity(city);
    };
    
    const handleUploadSuccess = () => {
        console.log("Upload completed, refreshing city panel data.");
    };

    return (
        <div className="min-h-screen pt-16 bg-gradient-to-br from-slate-950 via-purple-950 to-slate-950 text-white">
            <div className="max-w-full mx-auto p-0">

                {/* Map section */}
                <div 
                    className="relative z-10 w-full"//full screen map
                    style={{ height: 'calc(100vh - 64px)' }} // Full viewport height minus header
                >
                    {selectedCity === null && (
                        <div className="absolute top-4 w-full z-20 flex justify-center"> 
                            <div className="bg-slate-800/80 backdrop-blur-md p-3 px-8 rounded-full border border-purple-500/50 shadow-2xl flex items-center gap-3">
                                <MousePointerClick className="w-5 h-5 text-purple-400" />
                                <span className="text-sm md:text-base font-semibold text-white">
                                    Interactive Map: <span className="text-purple-300">Click a marker to explore & contribute!</span>
                                </span>
                            </div>
                        </div>
                    )}

                    <InteractiveMap 
                        onCitySelect={handleCitySelect} 
                    />
                </div>

                {/*Floating City Panel */}
                {selectedCity && (
                    <div 
                        className="fixed inset-0 z-30 flex items-center justify-center bg-black/60 backdrop-blur-sm"
                        onClick={() => setSelectedCity(null)} // Click dim background to close
                    >
                        <div 
                            className="bg-white rounded-3xl shadow-2xl max-h-[90vh] overflow-y-auto scrollbar-hidden 
                                        w-1/2  p-6 lg:p-10 relative"
                            onClick={(e) => e.stopPropagation()} // Prevent closing when clicking inside the panel
                        >

                {/* Panel Header */}
               <div className="mb-8 flex justify-between items-center border-b pb-4">
                 <div className="flex items-center gap-4">
                   <div className="flex-none w-14 h-14 bg-gradient-to-r from-purple-600 to-indigo-600 rounded-lg shadow-xl flex items-center justify-center">
                     <Network className="w-8 h-8 text-white" /> 
                 </div>
               <div>
            <h1 className="text-xl font-bold text-gray-900 leading-none">
                Community Hub!
            </h1>
            <p className="text-sm text-gray-500 mt-0.5">
                Local Accent Contribution Center
            </p>
        </div>
    </div>
    
    {/* Close Button */}
    <button 
        onClick={() => setSelectedCity(null)}
        className="flex-none p-2 text-gray-500 hover:text-red-600 transition-colors border border-gray-300 rounded-full"
        title="Close Panel"
    >
        <XCircle className="w-6 h-6" />
    </button>
</div>

                            
                            
                            <CityPanel 
                                city={selectedCity} 
                                onUploadSuccess={handleUploadSuccess} 
                            />{/* CityPanel component */}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

export default MapAndContributePage;