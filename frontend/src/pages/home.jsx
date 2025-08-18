import { useState, useEffect } from "react";
import { Mic, Globe, Users, Sparkles, ArrowRight, Play, Volume2 } from "lucide-react";

export default function Home() {
  const [isVisible, setIsVisible] = useState(false);
  const [currentAccent, setCurrentAccent] = useState(0);
  
  const accents = [
    { city: "Bangalore", region: "Karnataka", color: "from-blue-500 to-purple-600" },
    { city: "Mumbai", region: "Maharashtra", color: "from-green-500 to-teal-600" },
    { city: "Delhi", region: "NCR", color: "from-orange-500 to-red-600" },
    { city: "Panaji", region: "Goa", color: "from-purple-500 to-pink-600" },
  ];

  useEffect(() => {
    setIsVisible(true);
    const interval = setInterval(() => {
      setCurrentAccent((prev) => (prev + 1) % accents.length);
    }, 2500);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen text-white overflow-hidden">
      {/* Animatiion for background */}
      <div className="absolute inset-0">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-blue-500/10 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute bottom-1/4 right-1/4 w-80 h-80 bg-purple-500/10 rounded-full blur-3xl animate-pulse delay-1000"></div>
        <div className="absolute top-3/4 left-1/2 w-64 h-64 bg-pink-500/10 rounded-full blur-3xl animate-pulse delay-2000"></div>
      </div>

      
      <div className="relative z-10 pt-20">
        {/* Hero Section */}
        <div className="container mx-auto px-6 py-16">
          {/* Hero Content */}
          <div className={`text-center mb-16 transition-all duration-1000 ${isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-10'}`}>
            <div className="flex items-center justify-center mb-6">
              <div className="relative">
                <div className="absolute inset-0 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full blur-lg opacity-70 animate-pulse"></div>
                <Globe className="relative z-10 w-16 h-16 text-white" />
              </div>
            </div>
            
            <h1 className="text-6xl md:text-7xl font-extrabold mb-6 bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent leading-tight">
              Voice Analysis
            </h1>
            
            <div className="relative h-16 mb-8">
              <div className="absolute inset-0 flex items-center justify-center">
                {accents.map((accent, index) => (
                  <div
                    key={accent.city}
                    className={`absolute transition-all duration-700 ${
                      index === currentAccent
                        ? 'opacity-100 transform scale-100'
                        : 'opacity-0 transform scale-95'
                    }`}
                  >
                    <div className={`px-6 py-2 rounded-full bg-gradient-to-r ${accent.color} text-white font-semibold text-xl shadow-lg`}>
                      {accent.city}, {accent.region}
                    </div>
                  </div>
                ))}
              </div>
            </div>
            
            <p className="text-xl md:text-2xl text-gray-300 max-w-4xl mx-auto leading-relaxed">
              Discover the linguistic diversity across Indian regions through AI-powered accent analysis. 
              Record your voice and uncover the unique patterns that make your speech distinctly yours.
            </p>
          </div>

          
          <div className={`grid md:grid-cols-3 gap-8 mb-16 transition-all duration-1000 delay-300 ${isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-10'}`}>
            <div className="group bg-white/5 backdrop-blur-lg rounded-2xl p-8 border border-white/10 hover:border-blue-500/50 transition-all duration-300 hover:transform hover:scale-105">
              <div className="w-16 h-16 bg-gradient-to-r from-blue-500 to-purple-600 rounded-2xl flex items-center justify-center mb-6 group-hover:rotate-12 transition-transform duration-300">
                <Mic className="w-8 h-8 text-white" />
              </div>
              <h3 className="text-2xl font-bold mb-4 text-white">Record & Analyze</h3>
              <p className="text-gray-300 leading-relaxed">
                Simply speak into your device and let our advanced AI identify the subtle patterns in your accent.
              </p>
            </div>

            <div className="group bg-white/5 backdrop-blur-lg rounded-2xl p-8 border border-white/10 hover:border-purple-500/50 transition-all duration-300 hover:transform hover:scale-105">
              <div className="w-16 h-16 bg-gradient-to-r from-purple-500 to-pink-600 rounded-2xl flex items-center justify-center mb-6 group-hover:rotate-12 transition-transform duration-300">
                <Sparkles className="w-8 h-8 text-white" />
              </div>
              <h3 className="text-2xl font-bold mb-4 text-white">AI Prediction</h3>
              <p className="text-gray-300 leading-relaxed">
                Get detailed insights about your regional accent patterns and linguistic characteristics.
              </p>
            </div>

            <div className="group bg-white/5 backdrop-blur-lg rounded-2xl p-8 border border-white/10 hover:border-green-500/50 transition-all duration-300 hover:transform hover:scale-105">
              <div className="w-16 h-16 bg-gradient-to-r from-green-500 to-teal-600 rounded-2xl flex items-center justify-center mb-6 group-hover:rotate-12 transition-transform duration-300">
                <Users className="w-8 h-8 text-white" />
              </div>
              <h3 className="text-2xl font-bold mb-4 text-white">Explore Diversity</h3>
              <p className="text-gray-300 leading-relaxed">
                Discover how accents vary across different regions including Karnataka, Maharashtra, Delhi, and Goa.
              </p>
            </div>
          </div>

          
          <div className={`text-center transition-all duration-1000 delay-500 ${isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-10'}`}>
            <div className="bg-white/5 backdrop-blur-lg rounded-3xl p-12 border border-white/10 max-w-2xl mx-auto">
              <div className="flex items-center justify-center mb-6">
                <Volume2 className="w-12 h-12 text-blue-400 animate-pulse" />
              </div>
              
              <h2 className="text-3xl font-bold mb-6 text-white">
                Ready to discover your accent?
              </h2>
              
              <p className="text-gray-300 mb-8 text-lg">
                Join users exploring India's rich linguistic landscape across multiple regions
              </p>
              
              <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
                <a
                  href="/upload"
                  className="group relative px-8 py-4 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-xl font-semibold text-lg shadow-xl hover:shadow-2xl transition-all duration-300 hover:transform hover:scale-105 flex items-center gap-3"
                >
                  <Play className="w-5 h-5" />
                  Start Accent Analysis
                  <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform duration-300" />
                  <div className="absolute inset-0 bg-gradient-to-r from-blue-700 to-purple-700 rounded-xl opacity-0 group-hover:opacity-100 transition-opacity duration-300 -z-10"></div>
                </a>
                
                <button className="px-8 py-4 border-2 border-white/20 text-white rounded-xl font-semibold text-lg hover:border-white/40 hover:bg-white/5 transition-all duration-300 flex items-center gap-3">
                  <Mic className="w-5 h-5" />
                  Learn More
                </button>
              </div>
            </div>
          </div>

          {/* Stats Section */}
          <div className={`mt-20 grid grid-cols-2 md:grid-cols-4 gap-8 transition-all duration-1000 delay-700 ${isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-10'}`}>
            {[
              { number: "4", label: "Regions" },
              { number: "Karnataka", label: "Primary Focus" },
              { number: "Multi", label: "State Coverage" },
              { number: "95%", label: "Accuracy Rate" }
            ].map((stat, index) => (
              <div key={index} className="text-center">
                <div className="text-2xl md:text-3xl font-bold text-transparent bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text mb-2">
                  {stat.number}
                </div>
                <div className="text-gray-400 text-sm md:text-base">{stat.label}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}