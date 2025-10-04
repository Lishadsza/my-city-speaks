import React, { useState, useEffect } from 'react';
import { Globe, ArrowRight, Play, Mic, Users, Sparkles, Zap, MapPin, TrendingUp, Award } from 'lucide-react';

const HomePage = () => {
    const [isVisible, setIsVisible] = useState(false);
    const [currentAccent, setCurrentAccent] = useState(0);
    const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
    //animated accent thing
    const accents = [
        { city: "Bengaluru", region: "Karnataka", color: "from-blue-500 to-purple-600" },
        { city: "Mumbai", region: "Maharashtra", color: "from-green-500 to-teal-600" },
        { city: "Delhi", region: "NCR", color: "from-orange-500 to-red-600" },
        { city: "Chennai", region: "Tamil Nadu", color: "from-purple-500 to-pink-600" },
    ];

    useEffect(() => { //Effects for into page and changing of accenst
        setIsVisible(true);
        const interval = setInterval(() => {
            setCurrentAccent((prev) => (prev + 1) % accents.length);
        }, 3000);
        return () => clearInterval(interval);
    }, []);

    useEffect(() => {
        const handleMouseMove = (e) => {
            setMousePosition({ x: e.clientX, y: e.clientY });
        };
        window.addEventListener('mousemove', handleMouseMove);
        return () => window.removeEventListener('mousemove', handleMouseMove);
    }, []);
    return (
        <div className="min-h-screen bg-gradient-to-br from-slate-950 via-purple-950 to-slate-950 text-white overflow-hidden relative"> 
            {/* Animation for Background */}
            <div className="absolute inset-0 overflow-hidden">
                <div 
                    className="absolute w-[600px] h-[600px] bg-blue-500/20 rounded-full blur-[120px] transition-all duration-1000 animate-float"
                    style={{
                        top: `${20 + mousePosition.y * 0.02}%`,
                        left: `${20 + mousePosition.x * 0.02}%`,
                    }}
                ></div>
                <div className="absolute bottom-1/4 right-1/4 w-[500px] h-[500px] bg-purple-500/15 rounded-full blur-[100px] animate-float-delayed"></div>
                <div className="absolute top-3/4 left-1/2 w-[400px] h-[400px] bg-pink-500/15 rounded-full blur-[100px] animate-float-slow"></div>
                
                {/* Grid overlay with fade-in */}
                <div className="absolute inset-0 bg-[linear-gradient(to_right,#4f4f4f12_1px,transparent_1px),linear-gradient(to_bottom,#4f4f4f12_1px,transparent_1px)] bg-[size:4rem_4rem] opacity-0 animate-fade-in"></div>
            </div>

            {/* Add custom animations via style tag */}
            <style>{`
                @keyframes shine {
                    0% { transform: translateX(-100%) rotate(45deg); opacity: 0; }
                    50% { opacity: 0.3; }
                    100% { transform: translateX(100%) rotate(45deg); opacity: 0; }
                }
                @keyframes float {
                    0%, 100% { transform: translateY(0px) scale(1); }
                    50% { transform: translateY(-20px) scale(1.05); }
                }
                @keyframes float-delayed {
                    0%, 100% { transform: translateY(0px) scale(1); }
                    50% { transform: translateY(-30px) scale(1.08); }
                }
                @keyframes float-slow {
                    0%, 100% { transform: translateY(0px) scale(1); }
                    50% { transform: translateY(-15px) scale(1.03); }
                }
                @keyframes fade-in {
                    from { opacity: 0; }
                    to { opacity: 0.2; }
                }
                @keyframes slide-up {
                    from { 
                        opacity: 0; 
                        transform: translateY(30px); 
                    }
                    to { 
                        opacity: 1; 
                        transform: translateY(0); 
                    }
                }
                @keyframes slide-down {
                    from { 
                        opacity: 0; 
                        transform: translateY(-30px); 
                    }
                    to { 
                        opacity: 1; 
                        transform: translateY(0); 
                    }
                }
                @keyframes scale-in {
                    from { 
                        opacity: 0; 
                        transform: scale(0.8); 
                    }
                    to { 
                        opacity: 1; 
                        transform: scale(1); 
                    }
                }
                .animate-shine::before {
                    content: '';
                    position: absolute;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
                    animation: shine 2s ease-in-out;
                }
                .animate-float {
                    animation: float 6s ease-in-out infinite;
                }
                .animate-float-delayed {
                    animation: float-delayed 8s ease-in-out infinite;
                    animation-delay: 1s;
                }
                .animate-float-slow {
                    animation: float-slow 10s ease-in-out infinite;
                    animation-delay: 2s;
                }
                .animate-fade-in {
                    animation: fade-in 2s ease-out forwards;
                }
                .animate-slide-up {
                    animation: slide-up 0.8s ease-out forwards;
                }
                .animate-slide-down {
                    animation: slide-down 0.8s ease-out forwards;
                }
                .animate-scale-in {
                    animation: scale-in 0.6s ease-out forwards;
                }
            `}</style>

            <div className="relative z-10 pt-20"> 
                {/* Hero Section */}
                <div className="container mx-auto px-6 py-20">
                    <div className={`text-center mb-20 transition-all duration-1000 ${isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-10'}`}>
                        
                        {/* Icon with glow effect - animated */}
                        <div className="flex items-center justify-center mb-8 opacity-0 animate-scale-in" style={{animationDelay: '0.2s'}}>
                            <div className="relative">
                                <div className="absolute inset-0 bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500 rounded-full blur-2xl opacity-60 animate-pulse"></div>
                                <div className="relative w-24 h-24 bg-gradient-to-br from-blue-500 via-purple-600 to-pink-600 rounded-full flex items-center justify-center shadow-2xl border-4 border-white/10">
                                    <Zap className="w-12 h-12 text-white" />
                                </div>
                            </div>
                        </div>
                        
                        {/* Main Title with staggered animation */}
                        <h1 className="text-5xl md:text-7xl lg:text-8xl font-black mb-4 leading-[1.1] tracking-tight">
                            <span className="block bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent pb-2 opacity-0 animate-slide-down" style={{animationDelay: '0.4s'}}>
                                Discover India's
                            </span>
                            <span className="block text-white opacity-0 animate-slide-down" style={{animationDelay: '0.6s'}}>
                                Linguistic Heritage
                            </span>
                        </h1>

                        <p className="text-lg md:text-xl text-blue-200 font-medium mb-12 tracking-wide uppercase opacity-0 animate-fade-in" style={{animationDelay: '0.8s'}}>
                            Map • Explore • Contribute
                        </p>
                        
                        {/*  effects on accent with animations */}
                        <div className="relative h-20 mb-12 opacity-0 animate-slide-up" style={{animationDelay: '1.0s'}}>
                            <div className="absolute inset-0 flex items-center justify-center">
                                {accents.map((accent, index) => (
                                    <div
                                        key={accent.city}
                                        className={`absolute transition-all duration-1000 ease-in-out ${
                                            index === currentAccent
                                                ? 'opacity-100 transform scale-100 rotate-0'
                                                : 'opacity-0 transform scale-75 rotate-3'
                                        }`}
                                    >
                                        <div className={`relative px-10 py-3 rounded-full bg-gradient-to-r ${accent.color} shadow-2xl border border-white/30 backdrop-blur-md`}>
                                            <div className="absolute inset-0 overflow-hidden rounded-full">{/* Shine effect annimation */}
                                                <div className={`absolute inset-0 ${index === currentAccent ? 'animate-shine' : ''}`}></div>
                                            </div>
                                            <div className="relative text-center whitespace-nowrap">
                                                <span className="text-xl font-bold text-white">{accent.city}</span>
                                                <span className="text-xl text-white/80 font-medium">, {accent.region}</span>
                                            </div>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                        
                        <p className="text-xl md:text-2xl text-gray-300 max-w-4xl mx-auto leading-relaxed mb-14 font-light opacity-0 animate-slide-up" style={{animationDelay: '1.2s'}}>
                            Explore the rich tapestry of Indian accents through our AI-powered platform. 
                            <span className="text-purple-300 font-medium"> Record, analyze, and discover</span> the unique voices that make India diverse.
                        </p>

                        {/* button annimations  */}
                        <div className="flex flex-col sm:flex-row gap-6 justify-center items-center opacity-0 animate-slide-up" style={{animationDelay: '1.4s'}}>
                            
                            <a
                                href="/map"
                                className="group px-12 py-6 bg-white/5 backdrop-blur-md border-2 border-white/20 text-white rounded-2xl font-bold text-lg hover:bg-white/10 hover:border-green-400/50 transition-all duration-300 flex items-center gap-4 hover:shadow-xl hover:shadow-green-500/20"
                            >
                                <Globe className="w-7 h-7 group-hover:scale-110 transition-transform duration-300" />
                                <span>Explore Interactive Map</span>
                            </a>
                            
                            <a 
                                href="/analysis"
                                className="group px-12 py-6 bg-white/5 backdrop-blur-md border-2 border-white/20 text-white rounded-2xl font-bold text-lg hover:bg-white/10 hover:border-purple-400/50 transition-all duration-300 flex items-center gap-4 hover:shadow-xl hover:shadow-purple-500/20"
                            >
                                <Play className="w-7 h-7 group-hover:scale-110 transition-transform duration-300" />
                                <span>Analyze Your Accent</span>
                            </a>
                        </div>
                    </div>

                    {/* Features  */}
                    <div className={`grid md:grid-cols-3 gap-8 mb-20 transition-all duration-1000 delay-300 ${isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-10'}`}>
                        <div className="group relative bg-gradient-to-br from-white/10 to-white/5 backdrop-blur-xl rounded-3xl p-8 border border-white/20 hover:border-blue-400/50 transition-all duration-500 hover:transform hover:-translate-y-3 hover:shadow-2xl hover:shadow-blue-500/30 overflow-hidden">
                            <div className="absolute top-0 right-0 w-32 h-32 bg-blue-500/10 rounded-full blur-3xl group-hover:scale-150 transition-transform duration-500"></div>
                            <div className="relative">
                                <div className="w-20 h-20 bg-gradient-to-br from-blue-500 to-purple-600 rounded-2xl flex items-center justify-center mb-6 group-hover:rotate-12 group-hover:scale-110 transition-all duration-500 shadow-lg">
                                    <Mic className="w-10 h-10 text-white" />
                                </div>
                                <h3 className="text-2xl font-bold mb-4 text-white">Crowd-Sourced Audio</h3>
                                <p className="text-gray-300 leading-relaxed mb-4">
                                    Contribute your voice to build India's most comprehensive accent database. Every recording helps preserve linguistic diversity.
                                </p>
                                <div className="flex items-center gap-2 text-blue-400 font-semibold">
                                    <MapPin className="w-4 h-4" />
                                    <span className="text-sm">Location-based Recording</span>
                                </div>
                            </div>
                        </div>

                        <div className="group relative bg-gradient-to-br from-white/10 to-white/5 backdrop-blur-xl rounded-3xl p-8 border border-white/20 hover:border-purple-400/50 transition-all duration-500 hover:transform hover:-translate-y-3 hover:shadow-2xl hover:shadow-purple-500/30 overflow-hidden">
                            <div className="absolute top-0 right-0 w-32 h-32 bg-purple-500/10 rounded-full blur-3xl group-hover:scale-150 transition-transform duration-500"></div>
                            <div className="relative">
                                <div className="w-20 h-20 bg-gradient-to-br from-purple-500 to-pink-600 rounded-2xl flex items-center justify-center mb-6 group-hover:rotate-12 group-hover:scale-110 transition-all duration-500 shadow-lg">
                                    <Sparkles className="w-10 h-10 text-white" />
                                </div>
                                <h3 className="text-2xl font-bold mb-4 text-white">AI-Powered Analysis</h3>
                                <p className="text-gray-300 leading-relaxed mb-4">
                                    Advanced machine learning models analyze accent patterns, phonetic variations, and regional characteristics with high accuracy.
                                </p>
                                <div className="flex items-center gap-2 text-purple-400 font-semibold">
                                    <TrendingUp className="w-4 h-4" />
                                    <span className="text-sm">95% Accuracy Rate</span>
                                </div>
                            </div>
                        </div>

                        <div className="group relative bg-gradient-to-br from-white/10 to-white/5 backdrop-blur-xl rounded-3xl p-8 border border-white/20 hover:border-green-400/50 transition-all duration-500 hover:transform hover:-translate-y-3 hover:shadow-2xl hover:shadow-green-500/30 overflow-hidden">
                            <div className="absolute top-0 right-0 w-32 h-32 bg-green-500/10 rounded-full blur-3xl group-hover:scale-150 transition-transform duration-500"></div>
                            <div className="relative">
                                <div className="w-20 h-20 bg-gradient-to-br from-green-500 to-teal-600 rounded-2xl flex items-center justify-center mb-6 group-hover:rotate-12 group-hover:scale-110 transition-all duration-500 shadow-lg">
                                    <Users className="w-10 h-10 text-white" />
                                </div>
                                <h3 className="text-2xl font-bold mb-4 text-white">Community Driven</h3>
                                <p className="text-gray-300 leading-relaxed mb-4">
                                    Join a growing community of language enthusiasts. Fulfill requests, explore contributions, and connect through voices.
                                </p>
                                <div className="flex items-center gap-2 text-green-400 font-semibold">
                                    <Award className="w-4 h-4" />
                                    <span className="text-sm">Growing Community</span>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Stats Section */}
                    <div className={`transition-all duration-1000 delay-500 ${isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-10'}`}>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-6 max-w-5xl mx-auto">
                            {[
                                
                                { number: "20+", label: "Mapped City Hubs" },
                                { number: "95%", label: "Model Confidence" },
                                 { number: "25+", label: "Sample Phrases" },
                                { number: "4", label: "Regions for Analysis " }
                            ].map((stat, index) => (
                                <div key={index} className="group bg-white/5 backdrop-blur-lg rounded-2xl p-6 border border-white/10 hover:border-purple-400/50 transition-all duration-300 hover:scale-105 text-center">
                                    <div className="text-4xl md:text-5xl font-black bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent mb-2">
                                        {stat.number}
                                    </div>
                                    <div className="text-gray-400 text-sm md:text-base font-medium">{stat.label}</div>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            </div>
            <footer className="relative z-10 mt-32 border-t border-white/10 bg-gradient-to-b from-transparent to-black/30 backdrop-blur-xl">
                <div className="container mx-auto px-6 py-12">
                    <div className="text-center text-gray-400 text-sm">
                        <p>&copy; {new Date().getFullYear()} My City Speaks. Preserving linguistic diversity through technology.</p>
                    </div>
                </div>
            </footer>
        </div>
    );
};

export default HomePage;