import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Home, Map, BarChart2, Mail } from 'lucide-react'; // Changed Mic to Home icon

const navItems = [
    // 1. Home link
    { name: 'Home', path: '/', icon: Home }, 
    // 2. Map link
    { name: 'Map', path: '/map', icon: Map },
    // 3. Analysis link 
    { name: 'Analysis', path: '/analysis', icon: BarChart2 },
    // 4. Contact Me link
    { name: 'Contact Me', path: '/contact', icon: Mail },
];

const Navbar = () => {
    const location = useLocation();

    return (
        <header className="fixed top-0 left-0 right-0 z-50 bg-slate-900/80 backdrop-blur-xl border-b border-white/10 shadow-lg">
            <div className="max-w-7xl mx-auto px-6">
                <div className="flex justify-between items-center h-16">
                    
                    {/* Logo */}
                    <Link to="/" className="text-2xl font-bold text-white flex items-center gap-2 hover:text-purple-400 transition-colors">
                        <Map className="w-6 h-6 text-purple-400" /> {/* Changed icon to map for branding */}
                        My City Speaks
                    </Link>

                    {/* Navigation Links */}
                    <nav className="hidden sm:flex space-x-6">
                        {navItems.map((item) => (
                            <Link
                                key={item.name}
                                to={item.path}
                                className={`flex items-center gap-2 px-3 py-2 rounded-lg font-medium transition-all duration-300 ${
                                    location.pathname === item.path
                                        ? 'bg-purple-600 text-white shadow-md'
                                        : 'text-gray-300 hover:text-white hover:bg-white/10'
                                }`}
                            >
                                <item.icon className="w-5 h-5" />
                                {item.name}
                            </Link>
                        ))}
                    </nav>
                </div>
            </div>
        </header>
    );
};

export default Navbar;