import { useState, useEffect } from "react";
import { Mail, Globe, Send, MapPin, MessageCircle, AlertTriangle, CheckCircle } from "lucide-react";

export default function Contact() {
    const [isVisible, setIsVisible] = useState(false);
    const [formData, setFormData] = useState({
        name: '',
        email: '',
        subject: '',
        message: ''
    });
    // State for form feedback (replaces alert/confirm)
    const [feedback, setFeedback] = useState(null);

    useEffect(() => {
        setIsVisible(true);
    }, []);

    const handleInputChange = (e) => {
        setFormData({
            ...formData,
            [e.target.name]: e.target.value
        });
    };

    const handleSubmit = () => {
        setFeedback(null);
        
        if (formData.name && formData.email && formData.subject && formData.message) {
            console.log('Form submitted:', formData);
            
            // SUCCESS FEEDBACK
            setFeedback({
                type: 'success',
                message: 'Thank you! Your message has been sent successfully.'
            });

            setFormData({ name: '', email: '', subject: '', message: '' });
            setTimeout(() => setFeedback(null), 5000);

        } else {
            // ERROR FEEDBACK
            setFeedback({
                type: 'error',
                message: 'Please ensure all fields are filled out before submitting.'
            });
            setTimeout(() => setFeedback(null), 5000);
        }
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-slate-950 via-purple-950 to-slate-950 text-white relative overflow-hidden">
            
            {/* Background elements */}
            <div className="absolute inset-0">
                <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-blue-500/10 rounded-full blur-3xl animate-pulse"></div>
                <div className="absolute bottom-1/4 right-1/4 w-80 h-80 bg-purple-500/10 rounded-full blur-3xl animate-pulse delay-1000"></div>
                <div className="absolute top-3/4 left-1/2 w-64 h-64 bg-pink-500/10 rounded-full blur-3xl animate-pulse delay-2000"></div>
            </div>

            <div className="relative z-10 pt-20">
                <div className="max-w-6xl mx-auto px-6 py-16">
                    {/* Header Section */}
                    <div className={`text-center mb-16 transition-all duration-1000 ${isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-10'}`}>
                        <div className="flex items-center justify-center mb-6">
                            <div className="relative">
                                <div className="absolute inset-0 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full blur-lg opacity-70 animate-pulse"></div>
                                <Globe className="relative z-10 w-16 h-16 text-white" />
                            </div>
                        </div>
                        
                        <h1 className="text-5xl md:text-6xl font-extrabold mb-6 bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent leading-tight">
                            Get In Touch
                        </h1>
                        
                        <div className="inline-block bg-blue-500/20 backdrop-blur-sm border border-blue-400/30 rounded-full px-6 py-3 mb-8">
                            <span className="text-blue-200 font-medium">Let's Connect & Collaborate</span>
                        </div>
                        
                        <p className="text-xl md:text-2xl text-gray-300 max-w-4xl mx-auto leading-relaxed">
                            Have questions about voice analysis or want to collaborate? I'd love to hear from you. 
                            Reach out and let's explore the fascinating world of linguistic patterns together.
                        </p>
                    </div>

                    {/*  Feedback Alert Box */}
                    {feedback && (
                        <div 
                            className={`max-w-2xl mx-auto p-4 rounded-xl shadow-xl mb-8 flex items-center transition-opacity duration-300 
                            ${feedback.type === 'success' ? 'bg-green-600/20 border border-green-500 text-green-300' : 'bg-red-600/20 border border-red-500 text-red-300'}`}
                        >
                            {feedback.type === 'success' ? <CheckCircle className="w-6 h-6 mr-3" /> : <AlertTriangle className="w-6 h-6 mr-3" />}
                            <p className="font-medium">{feedback.message}</p>
                        </div>
                    )}
                    

                    <div className="grid lg:grid-cols-3 gap-8 mb-16">
                        {/* Contact Information  */}
                        <div className="lg:col-span-1 space-y-6">
                            
                            {/* Email  */}
                            <div className="group bg-white/5 backdrop-blur-lg rounded-2xl p-6 border border-white/10 hover:border-blue-500/50 transition-all duration-300 hover:transform hover:scale-105 shadow-xl">
                                <div className="w-12 h-12 bg-gradient-to-r from-blue-500 to-purple-600 rounded-2xl flex items-center justify-center mb-4 group-hover:rotate-12 transition-transform duration-300 shadow-lg">
                                    <Mail className="w-6 h-6 text-white" />
                                </div>
                                <h3 className="text-xl font-bold mb-2 text-white">Email</h3>
                                <a 
                                    href="mailto:dsouzalisha24@gmail.com"
                                    className="text-blue-300 hover:text-blue-200 transition-colors break-all"
                                >
                                    dsouzalisha24@gmail.com
                                </a>
                            </div>

                            {/* Location */}
                            <div className="group bg-white/5 backdrop-blur-lg rounded-2xl p-6 border border-white/10 hover:border-green-500/50 transition-all duration-300 hover:transform hover:scale-105 shadow-xl">
                                <div className="w-12 h-12 bg-gradient-to-r from-green-500 to-teal-600 rounded-2xl flex items-center justify-center mb-4 group-hover:rotate-12 transition-transform duration-300 shadow-lg">
                                    <MapPin className="w-6 h-6 text-white" />
                                </div>
                                <h3 className="text-xl font-bold mb-2 text-white">Location</h3>
                                <p className="text-green-300">Karnataka, India</p>
                                <p className="text-gray-400 text-sm mt-1">Available for remote collaboration</p>
                            </div>

                            {/* WhatsApp  */}
                            <div className="group bg-white/5 backdrop-blur-lg rounded-2xl p-6 border border-white/10 hover:border-orange-500/50 transition-all duration-300 hover:transform hover:scale-105 shadow-xl">
                                <div className="w-12 h-12 bg-gradient-to-r from-orange-500 to-red-600 rounded-2xl flex items-center justify-center mb-4 group-hover:rotate-12 transition-transform duration-300 shadow-lg">
                                    <MessageCircle className="w-6 h-6 text-white" />
                                </div>
                                <h3 className="text-xl font-bold mb-2 text-white">WhatsApp</h3>
                                <a 
                                    href="https://wa.me/1234567890"
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="text-orange-300 hover:text-orange-200 transition-colors"
                                >
                                    +91 1234567890
                                </a>
                            </div>
                        </div>

                        {/* Contact Form */}
                        <div className="lg:col-span-2">
                            <div className="bg-white/5 backdrop-blur-lg rounded-3xl p-8 border border-white/10 shadow-2xl">
                                <div className="flex items-center space-x-3 mb-8">
                                    <div className="w-12 h-12 bg-gradient-to-r from-blue-500 to-purple-600 rounded-2xl flex items-center justify-center shadow-lg">
                                        <Send className="w-6 h-6 text-white" />
                                    </div>
                                    <h2 className="text-3xl font-bold text-white">Send a Message</h2>
                                </div>

                                <div className="space-y-6">
                                    <div className="grid md:grid-cols-2 gap-6">
                                        <div>
                                            <label className="block text-white/80 font-medium mb-2">
                                                Your Name
                                            </label>
                                            <input
                                                type="text"
                                                name="name"
                                                value={formData.name}
                                                onChange={handleInputChange}
                                               
                                                className="w-full bg-white/10 border border-white/20 rounded-xl px-4 py-3 text-white placeholder-white/50 focus:outline-none focus:border-blue-500/50 focus:bg-white/15 transition-all duration-300"
                                                placeholder="Enter your name"
                                            />
                                        </div>
                                        
                                        <div>
                                            <label className="block text-white/80 font-medium mb-2">
                                                Email Address
                                            </label>
                                            <input
                                                type="email"
                                                name="email"
                                                value={formData.email}
                                                onChange={handleInputChange}
                                                className="w-full bg-white/10 border border-white/20 rounded-xl px-4 py-3 text-white placeholder-white/50 focus:outline-none focus:border-blue-500/50 focus:bg-white/15 transition-all duration-300"
                                                placeholder="your.email@example.com"
                                            />
                                        </div>
                                    </div>

                                    <div>
                                        <label className="block text-white/80 font-medium mb-2">
                                            Subject
                                        </label>
                                        <input
                                            type="text"
                                            name="subject"
                                            value={formData.subject}
                                            onChange={handleInputChange}
                                            className="w-full bg-white/10 border border-white/20 rounded-xl px-4 py-3 text-white placeholder-white/50 focus:outline-none focus:border-blue-500/50 focus:bg-white/15 transition-all duration-300"
                                            placeholder="What's this about?"
                                        />
                                    </div>

                                    <div>
                                        <label className="block text-white/80 font-medium mb-2">
                                            Message
                                        </label>
                                        <textarea
                                            name="message"
                                            value={formData.message}
                                            onChange={handleInputChange}
                                            rows={6}
                                            className="w-full bg-white/10 border border-white/20 rounded-xl px-4 py-3 text-white placeholder-white/50 focus:outline-none focus:border-blue-500/50 focus:bg-white/15 transition-all duration-300 resize-none"
                                            placeholder="Tell me about your project, questions, or ideas..."
                                        />
                                    </div>

                                    <div className="flex flex-col sm:flex-row gap-4">
                                        <button
                                            onClick={handleSubmit}
                                            
                                            className="group flex-1 relative px-8 py-4 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-xl font-semibold text-lg shadow-xl hover:shadow-2xl transition-all duration-300 hover:transform hover:scale-105 flex items-center justify-center gap-3"
                                        >
                                            <Send className="w-5 h-5 group-hover:translate-x-1 group-hover:-translate-y-1 transition-transform duration-300" />
                                            Send Message
                                            <div className="absolute inset-0 bg-gradient-to-r from-blue-700 to-purple-700 rounded-xl opacity-0 group-hover:opacity-100 transition-opacity duration-300 -z-10"></div>
                                        </button>
                                        
                                        <button
                                            onClick={() => setFormData({ name: '', email: '', subject: '', message: '' })}
                                            className="px-8 py-4 border-2 border-white/20 text-white rounded-xl font-semibold text-lg hover:border-white/40 hover:bg-white/5 transition-all duration-300"
                                        >
                                            Clear Form
                                        </button>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}