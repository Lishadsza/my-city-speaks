import { Routes, Route, Link } from 'react-router-dom';
import { Home as HomeIcon, Upload, Phone, Globe } from 'lucide-react';
import Home from './pages/home.jsx';
import UploadPage from './pages/UploadPage.jsx';
import Contact from './pages/contact.jsx';

export default function App() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-900 via-purple-800 to-pink-700">
      {/* Fixed Navigation Header */}
      <nav className="fixed top-0 w-full z-50 bg-slate-800/80 backdrop-blur-sm border-b border-white/10">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            {/* Left side - Brand */}
            <Link to="/" className="flex items-center space-x-3 hover:opacity-80 transition-opacity">
              <div className="w-8 h-8 bg-white rounded-full flex items-center justify-center">
                <Globe className="w-5 h-5 text-purple-600" />
              </div>
              <span className="text-white font-semibold text-xl">My City Speaks</span>
            </Link>

            {/* Right side - Navigation Links */}
            <div className="flex items-center space-x-6">
              <Link
                to="/"
                className="flex items-center space-x-2 text-white/80 hover:text-white transition-colors"
              >
                <HomeIcon className="w-4 h-4" />
                <span>Home</span>
              </Link>
              <Link
                to="/upload"
                className="flex items-center space-x-2 text-white/80 hover:text-white transition-colors"
              >
                <Upload className="w-4 h-4" />
                <span>Analysis</span>
              </Link>
              <Link
                to="/contact"
                className="flex items-center space-x-2 text-white/80 hover:text-white transition-colors"
              >
                <Phone className="w-4 h-4" />
                <span>Contact</span>
              </Link>
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/upload" element={<UploadPage />} />
          <Route path="/contact" element={<Contact />} />
        </Routes>
      </main>
    </div>
  );
}

