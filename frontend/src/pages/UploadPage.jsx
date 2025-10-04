import React, { useState, useRef } from "react";
import { Upload, Mic, MicOff, RotateCcw, FileAudio, Globe, AlertTriangle } from "lucide-react";

export default function UploadPage() {
    const [uploading, setUploading] = useState(false);
    const [isRecording, setIsRecording] = useState(false);
    const [audioURL, setAudioURL] = useState("");
    const [audioBlob, setAudioBlob] = useState(null);
    const [prediction, setPrediction] = useState(null);
    const [selectedFile, setSelectedFile] = useState(null);
    const [statusMessage, setStatusMessage] = useState(null); // For state-driven feedback
    const mediaRecorderRef = useRef(null);
    const audioChunksRef = useRef([]);

    // Helper to display status message
    const setTempStatus = (message, type = 'error') => {
        setStatusMessage({ message, type });
        setTimeout(() => setStatusMessage(null), 4000);
    };

    // Reset
    const handleReset = () => {
        setUploading(false);
        setIsRecording(false);
        setAudioURL("");
        setAudioBlob(null);
        setPrediction(null);
        setSelectedFile(null);
        setStatusMessage(null);
        if (mediaRecorderRef.current) {
            mediaRecorderRef.current.stream.getTracks().forEach((track) => track.stop());
        }
    };

    // File Upload
    const handleFileChange = (e) => {
        const file = e.target.files[0];
        setSelectedFile(file || null);
    };

    const handleFileUpload = async () => {
        if (!selectedFile) return setTempStatus("Please select an audio file first.");
        await uploadToServer(selectedFile);
    };

    // Recording
    const startRecording = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            const recorder = new MediaRecorder(stream);
            mediaRecorderRef.current = recorder;
            audioChunksRef.current = [];

            recorder.ondataavailable = (event) => {
                if (event.data.size > 0) audioChunksRef.current.push(event.data);
            };

            recorder.onstop = () => {
                const blob = new Blob(audioChunksRef.current, { type: "audio/wav" });
                const url = URL.createObjectURL(blob);
                setAudioBlob(blob);
                setAudioURL(url);
            };

            recorder.start();
            setIsRecording(true);
        } catch (err) {
            console.error("Mic access denied:", err);
            setTempStatus("Microphone access denied. Please check browser permissions.", 'error');
        }
    };

    const stopRecording = () => {
        mediaRecorderRef.current.stop();
        setIsRecording(false);
    };

    const uploadRecording = async () => {
        if (!audioBlob) return setTempStatus("No recording detected to analyze.");
        await uploadToServer(audioBlob, "recording.wav");
    };

    // Common upload logic
    const uploadToServer = async (file, fileName) => {
        setUploading(true);
        setStatusMessage(null);
        const formData = new FormData();
        formData.append("file", file, fileName || file.name);

        try {
            const res = await fetch("http://127.0.0.1:5000/predict", {
                method: "POST",
                body: formData,
            });
            
            const data = await res.json();
            setPrediction(data);
            
            if (data.error) {
                setTempStatus(`Analysis Failed: ${data.error}`, 'error');
            }
            
        } catch (err) {
            console.error("Upload error", err);
            setPrediction({ error: "Connection Error: Failed to reach Flask server." });
            setTempStatus("Upload failed. Check your network and Flask server status.", 'error');
        }
        setUploading(false);
    };

    return (
        //  Dark gradient background
        <div className="min-h-screen bg-gradient-to-br from-slate-950 via-purple-950 to-slate-950 text-white relative overflow-hidden">
            
            <div className="relative z-10 pt-20">
                <div className="max-w-4xl mx-auto px-6 py-16">
                    
                    {/* Header Section */}
                    <div className="text-center mb-16">
                        <div className="w-16 h-16 bg-white/10 backdrop-blur-sm rounded-full flex items-center justify-center mx-auto mb-6 shadow-xl">
                            <Globe className="w-8 h-8 text-white" />
                        </div>
                        <h1 className="text-5xl md:text-6xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent mb-6">
                            Voice Analysis
                        </h1>
                        <div className="inline-block bg-blue-500/20 backdrop-blur-sm border border-blue-400/30 rounded-full px-6 py-3 mb-8">
                            <span className="text-blue-200 font-medium">AI-Powered Accent Detection</span>
                        </div>
                        <p className="text-xl text-gray-300 max-w-3xl mx-auto leading-relaxed">
                            Upload your audio file or record your voice to discover the linguistic patterns
                            and regional accents that make your speech unique.
                        </p>
                    </div>
                    
                    {/* Status Message */}
                    {statusMessage && (
                        <div className={`p-4 mb-8 rounded-xl flex items-center ${statusMessage.type === 'error' ? 'bg-red-900/50 border-red-600 text-red-300' : 'bg-green-900/50 border-green-600 text-green-300'}`}>
                            <AlertTriangle className="w-5 h-5 mr-3"/>
                            <p className="font-medium">{statusMessage.message}</p>
                        </div>
                    )}
                    <div className="grid md:grid-cols-2 gap-8 mb-12">
                        {/* File Upload Card */}
                        <div className="bg-white/10 backdrop-blur-md rounded-3xl p-8 border border-white/20 hover:bg-white/15 transition-all duration-300 shadow-2xl">
                            <div className="flex items-center space-x-3 mb-6">
                                <div className="w-12 h-12 bg-blue-500/20 rounded-2xl flex items-center justify-center shadow-lg">
                                    <Upload className="w-6 h-6 text-blue-300" />
                                </div>
                                <h2 className="text-2xl font-semibold text-white">Upload Audio File</h2>
                            </div>

                            <div className="space-y-6">
                                <div className="relative">
                                    <input
                                        type="file"
                                        accept=".wav"
                                        onChange={handleFileChange}
                                        className="hidden"
                                        id="file-upload"
                                    />
                                    <label
                                        htmlFor="file-upload"
                                        className="flex items-center justify-center w-full h-32 border-2 border-dashed border-white/30 rounded-xl cursor-pointer hover:border-white/50 transition-colors group"
                                    >
                                        <div className="text-center">
                                            <FileAudio className="w-8 h-8 text-white/60 mx-auto mb-2 group-hover:text-white/80" />
                                            <p className="text-white/80 font-medium">
                                                {selectedFile ? selectedFile.name : "Click to select audio file"}
                                            </p>
                                            <p className="text-white/50 text-sm mt-1">.WAV files only</p>
                                        </div>
                                    </label>
                                </div>

                                <div className="flex space-x-3">
                                    <button
                                        onClick={handleFileUpload}
                                        disabled={uploading || !selectedFile}
                                        className="flex-1 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white py-3 px-6 rounded-xl font-medium transition-colors flex items-center justify-center space-x-2 shadow-lg hover:shadow-blue-500/50"
                                    >
                                        {uploading ? (
                                            <>
                                                <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                                                <span>Analyzing...</span>
                                            </>
                                        ) : (
                                            <>
                                                <Upload className="w-4 h-4" />
                                                <span>Analyze Audio</span>
                                            </>
                                        )}
                                    </button>
                                    <button
                                        onClick={handleReset}
                                        className="px-6 py-3 bg-white/10 hover:bg-white/20 text-white rounded-xl font-medium transition-colors border border-white/20"
                                    >
                                        <RotateCcw className="w-4 h-4" />
                                    </button>
                                </div>
                            </div>
                        </div>
                        {/* Recording Card */}
                        <div className="bg-white/10 backdrop-blur-md rounded-3xl p-8 border border-white/20 hover:bg-white/15 transition-all duration-300 shadow-2xl">
                            <div className="flex items-center space-x-3 mb-6">
                                <div className="w-12 h-12 bg-pink-500/20 rounded-2xl flex items-center justify-center shadow-lg">
                                    <Mic className="w-6 h-6 text-pink-300" />
                                </div>
                                <h2 className="text-2xl font-semibold text-white">Record Your Voice</h2>
                            </div>
                            <div className="space-y-6">
                                {/* Recording Controls */}
                                {!isRecording && !audioBlob && (
                                    <div className="text-center">
                                        <button
                                            onClick={startRecording}
                                            className="w-24 h-24 bg-green-600 hover:bg-green-700 rounded-full flex items-center justify-center mx-auto mb-4 transition-all duration-300 hover:scale-105 shadow-lg hover:shadow-green-500/50"
                                        >
                                            <Mic className="w-8 h-8 text-white" />
                                        </button>
                                        <p className="text-white/80">Click to start recording</p>
                                    </div>
                                )}

                                {isRecording && (
                                    <div className="text-center">
                                        <button
                                            onClick={stopRecording}
                                            className="w-24 h-24 bg-red-600 hover:bg-red-700 rounded-full flex items-center justify-center mx-auto mb-4 transition-all duration-300 animate-pulse shadow-lg shadow-red-500/50"
                                        >
                                            <MicOff className="w-8 h-8 text-white" />
                                        </button>
                                        <p className="text-red-300 font-medium">Recording... Click to stop</p>
                                    </div>
                                )}

                                {audioBlob && (
                                    <div className="space-y-4">
                                        <div className="bg-white/10 rounded-xl p-4 border border-white/20">
                                            <audio
                                                controls
                                                src={audioURL}
                                                className="w-full bg-transparent"
                                                style={{ filter: 'invert(0.9) hue-rotate(180deg) brightness(0.9)' }} 
                                            />
                                        </div>

                                        <div className="flex space-x-2">
                                            <button
                                                onClick={uploadRecording}
                                                disabled={uploading}
                                                className="flex-1 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white py-3 px-4 rounded-xl font-medium transition-colors flex items-center justify-center space-x-2 shadow-lg hover:shadow-blue-500/50"
                                            >
                                                {uploading ? (
                                                    <>
                                                        <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                                                        <span>Analyzing...</span>
                                                    </>
                                                ) : (
                                                    <>
                                                        <Upload className="w-4 h-4" />
                                                        <span>Analyze</span>
                                                    </>
                                                )}
                                            </button>
                                            <button
                                                onClick={() => {
                                                    setAudioBlob(null);
                                                    setAudioURL("");
                                                    startRecording();
                                                }}
                                                className="px-4 py-3 bg-white/10 hover:bg-white/20 text-white rounded-xl font-medium transition-colors border border-white/20"
                                            >
                                                <Mic className="w-4 h-4 text-pink-300" />
                                            </button>
                                            <button
                                                onClick={handleReset}
                                                className="px-4 py-3 bg-white/10 hover:bg-white/20 text-white rounded-xl font-medium transition-colors border border-white/20"
                                            >
                                                <RotateCcw className="w-4 h-4" />
                                            </button>
                                        </div>
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                    {/* Prediction Results */}
                    {prediction && (
                        <div className="bg-white/10 backdrop-blur-md rounded-3xl p-8 border border-white/20 max-w-2xl mx-auto shadow-2xl">
                            <div className="flex items-center space-x-3 mb-6">
                                <div className="w-12 h-12 bg-green-500/20 rounded-2xl flex items-center justify-center shadow-lg">
                                    <Globe className="w-6 h-6 text-green-300" />
                                </div>
                                <h3 className="text-2xl font-semibold text-white">Analysis Results</h3>
                            </div>

                            {prediction.error ? (
                                <div className="text-center py-8 bg-red-900/50 border border-red-600 rounded-xl">
                                    <div className="w-16 h-16 bg-red-500/20 rounded-full flex items-center justify-center mx-auto mb-4">
                                        <span className="text-2xl">⚠️</span>
                                    </div>
                                    <p className="text-red-300 text-lg">{prediction.error}</p>
                                </div>
                            ) : (
                                <div className="space-y-6">
                                    <div className="grid md:grid-cols-2 gap-6">
                                        <div className="bg-white/5 rounded-xl p-6 text-center border border-white/10">
                                            <p className="text-white/60 text-sm font-medium uppercase tracking-wide mb-2">Detected Language</p>
                                            <p className="text-white text-2xl font-bold">{prediction.language}</p>
                                        </div>

                                        {prediction.accent && (
                                            <div className="bg-white/5 rounded-xl p-6 text-center border border-white/10">
                                                <p className="text-white/60 text-sm font-medium uppercase tracking-wide mb-2">Regional Accent</p>
                                                <p className="text-white text-2xl font-bold">{prediction.accent}</p>
                                            </div>
                                        )}
                                    </div>

                                    {prediction.confidence !== undefined && !isNaN(prediction.confidence) && (
                                        <div className="bg-white/5 rounded-xl p-6 border border-white/10">
                                            <p className="text-white/60 text-sm font-medium uppercase tracking-wide mb-3">Confidence Level</p>
                                            <div className="flex items-center space-x-4">
                                                <div className="flex-1 bg-white/10 rounded-full h-3 overflow-hidden">
                                                    <div
                                                        className="h-full bg-gradient-to-r from-green-400 to-blue-500 rounded-full transition-all duration-1000"
                                                        style={{ width: `${prediction.confidence * 100}%` }}
                                                    ></div>
                                                </div>
                                                <span className="text-white text-xl font-bold min-w-16">
                                                    {(prediction.confidence * 100).toFixed(1)}%
                                                </span>
                                            </div>
                                        </div>
                                    )}

                                    {prediction.note && (
                                        <div className="bg-blue-500/10 border border-blue-400/20 rounded-xl p-6 shadow-inner">
                                            <p className="text-blue-200 italic">{prediction.note}</p>
                                        </div>
                                    )}
                                </div>
                            )}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}