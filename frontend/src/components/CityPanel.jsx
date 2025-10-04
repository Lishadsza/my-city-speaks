import React, { useState, useEffect, useRef } from 'react';
import { Link, useLocation, Routes, Route, useNavigate } from 'react-router-dom';
import { Home, Map, BarChart2, Mail, Loader, AlertTriangle, MapPin, XCircle, ChevronLeft, MousePointerClick, Send, Mic, Play, StopCircle, RotateCcw, BookOpen, MessageSquare, PlusCircle, UserCheck, Pause, Globe, CheckCircle } from 'lucide-react';

const FLASK_API_URL = 'http://127.0.0.1:5000'; 
const INDIA_CENTER = [22.3511, 78.6677]; 

const CITY_LANGUAGES_MAP = {
    "Delhi": ["Hindi"],
    "Mumbai": ["Marathi"],
    "Bengaluru": ["Kannada"],
    "Chennai": ["Tamil"],
    "Hyderabad": ["Telugu"],
    "Kolkata": ["Bengali"],
    "Mangaluru": ["Konkani", "Tulu", "Beary", "Kannada"], 
    "default": ["Hindi", "English", "Other"]
};




// Helper function to group recordings by language 
const groupRecordingsByLanguage = (recordings) => {
    if (!recordings) return {};
    return recordings.reduce((acc, recording) => {
        const lang = recording.language ? recording.language.trim() : 'Unknown Language';
        if (!acc[lang]) {
            acc[lang] = [];
        }
        acc[lang].push(recording);
        return acc;
    }, {});
};


//core logic
const CityPanel = ({ city, onUploadSuccess }) => {
    // State for Data Fetching 
    const [recordings, setRecordings] = useState([]);
    const [phrasesList, setPhrasesList] = useState([]); 
    const [pendingRequests, setPendingRequests] = useState([]); 
    const [isRecordingsLoading, setIsRecordingsLoading] = useState(false);
    const [isRequestsLoading, setIsRequestsLoading] = useState(false); 
    const [fetchError, setFetchError] = useState(null);
    
    //  State for Upload Form
    const [phraseId, setPhraseId] = useState(''); 
    const [language, setLanguage] = useState(''); 
    const [isUploading, setIsUploading] = useState(false);
    const [uploadStatus, setUploadStatus] = useState(null); 
    
    // Request Recording State 
    const [recordingRequestId, setRecordingRequestId] = useState(null); 
    const [recordingRequestText, setRecordingRequestText] = useState(null); 

    //  State for MediaRecorder
    const [isRecording, setIsRecording] = useState(false);
    const [recordedBlob, setRecordedBlob] = useState(null);
    const mediaRecorderRef = useRef(null);
    const audioChunksRef = useRef([]);

    //  State for Phrase Request Feature 
    const [requestPhraseText, setRequestPhraseText] = useState('');
    const [isRequesting, setIsRequesting] = useState(false);
    const [phraseRequestStatus, setPhraseRequestStatus] = useState(null);

    //  State for Custom Playback 
    const [playingAudioId, setPlayingAudioId] = useState(null);
    const audioRefs = useRef({}); 
    // Helper function (replaces alerts and sets status messages)
    const setTempStatus = (message) => { /* ... */ };
    
    
    const availableLanguages = CITY_LANGUAGES_MAP[city.name.trim()] || CITY_LANGUAGES_MAP['default'];

    //  Playback Control Functions
    const togglePlayback = (id) => {
        const audio = audioRefs.current[id];

        if (!audio) {
            console.error(`Audio element for ID ${id} not found.`);
            return;
        }

        if (playingAudioId === id) {
            // Stop current playback
            audio.pause();
            audio.currentTime = 0;
            setPlayingAudioId(null);
        } else {
            // Stop any other currently playing audio
            if (playingAudioId && audioRefs.current[playingAudioId]) {
                audioRefs.current[playingAudioId].pause();
                audioRefs.current[playingAudioId].currentTime = 0;
            }

            // Start new playback
            audio.play().catch(e => console.error("Playback failed:", e));
            setPlayingAudioId(id);
        }
        
        // Listen for 'ended' event to reset state
        audio.onended = () => setPlayingAudioId(null);
    };


    
    //  Fetch All Existing Phrases for the Dropdown
    const fetchPhrases = async () => {
        try {
            const response = await fetch(`${FLASK_API_URL}/api/phrases`);
            if (!response.ok) throw new Error("Failed to fetch phrase list.");
            const data = await response.json();
            setPhrasesList(data);
            // Auto-select first phrase if available
            if (data.length > 0 && !phraseId) {
                setPhraseId(data[0].id);
            }
        } catch (error) {
            console.error("Phrase fetch error:", error);
        }
    };
    
    //  Fetch City-Specific Recordings
    const fetchRecordings = async (cityId) => {
        setIsRecordingsLoading(true);
        setFetchError(null);
        try {
            const response = await fetch(`${FLASK_API_URL}/api/city_recordings/${cityId}`);
            if (!response.ok) throw new Error(`Failed to fetch recordings (Status: ${response.status})`);
            
            const data = await response.json();
            
            const formattedData = data.map(rec => ({
                ...rec,
                english_text: rec.phrases.english_text.trim()
            }));
            setRecordings(formattedData);
        } catch (error) {
            console.error("Fetch Error:", error);
            setFetchError("Could not load recordings. Ensure Flask server and Supabase RLS are correct.");
        } finally {
            setIsRecordingsLoading(false);
        }
    };

    //  Fetch Pending Phrase Requests for the Community Board
    const fetchRequests = async (cityId) => {
        setIsRequestsLoading(true);
        try {
            const response = await fetch(`${FLASK_API_URL}/api/city_requests/${cityId}`);
            if (!response.ok) throw new Error("Failed to fetch requests.");
            const data = await response.json();
            setPendingRequests(data);
        } catch (error) {
            console.error("Request fetch error:", error);
        } finally {
            setIsRequestsLoading(false);
        }
    };


    useEffect(() => {
        // Stop all audio when switching cities
        if (playingAudioId && audioRefs.current[playingAudioId]) {
             audioRefs.current[playingAudioId].pause();
             audioRefs.current[playingAudioId].currentTime = 0;
             setPlayingAudioId(null);
        }

        fetchPhrases();
        if (city && city.id) {
            fetchRecordings(city.id);
            fetchRequests(city.id); 
            setLanguage(availableLanguages[0]);
        }
    }, [city.id]); 

    // Group recordings for display
    const groupedRecordings = groupRecordingsByLanguage(recordings);

   
    
    // Core logic for starting recording
    const startMicrophoneCapture = async () => {
        setRecordedBlob(null); 
        audioChunksRef.current = []; 
        setUploadStatus(null); 

        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            const options = { mimeType: 'audio/webm' };
            const recorder = new MediaRecorder(stream, options);

            recorder.ondataavailable = (event) => {
                audioChunksRef.current.push(event.data);
            };

            recorder.onstop = () => {
                const blob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
                setRecordedBlob(blob);
                stream.getTracks().forEach(track => track.stop());
                setIsRecording(false);
            };

            mediaRecorderRef.current = recorder;
            recorder.start();
            setIsRecording(true);
            return true;

        } catch (error) {
            // Using console.error and custom state for message box instead of alert()
            console.error("Recording error: Microphone access denied or failed.", error);
            setUploadStatus('mic_error'); 
            setTimeout(() => setUploadStatus(null), 5000);
            return false;
        }
    };

  
    const startStandardRecording = async () => {
        setRecordingRequestId(null);
        setRecordingRequestText(null);
        await startMicrophoneCapture();
    }
    
    // NEW: Recording start for a specific request
    const startRecordingForRequest = async (requestId, requestText) => {
        // Ensure no standard phrase is selected when fulfilling a request
        setPhraseId(''); 
        setRecordingRequestId(requestId);
        setRecordingRequestText(requestText);
        
        const started = await startMicrophoneCapture();
        if (!started) {
             setRecordingRequestId(null);
             setRecordingRequestText(null);
        }
    }


    const stopRecording = () => {
        if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
            mediaRecorderRef.current.stop();
        }
    };

    const resetRecording = () => {
        setRecordedBlob(null);
        setUploadStatus(null);
        setRecordingRequestId(null); // Clear request ID
        setRecordingRequestText(null); // Clear request text
        
        if (phrasesList.length > 0) {
            setPhraseId(phrasesList[0].id);
        }
    };

    //  Upload Submission Logic 
    const handleSubmit = async (e) => {
        if (e && e.preventDefault) e.preventDefault();
        
        let phraseTextToSend;
        
        //  Determine the phrase text based on context
        if (recordingRequestId) {
            // Context: Fulfilling a community request
            phraseTextToSend = recordingRequestText;
        } else {
            // Context: Standard upload using the main dropdown
            const selectedPhrase = phrasesList.find(p => p.id === phraseId);
            if (!selectedPhrase) {
                setUploadStatus('error');
                console.error("Submission failed: No phrase selected or found.");
                setTimeout(() => setUploadStatus(null), 3000);
                return;
            }
            phraseTextToSend = selectedPhrase.english_text;
        }

        if (!recordedBlob || !phraseTextToSend || !language || !city.id) {
            setUploadStatus('error');
            console.error("Submission failed: Missing audio, phrase, language, or city ID.");
            setTimeout(() => setUploadStatus(null), 3000);
            return;
        }

        setIsUploading(true);
        setUploadStatus(null);
        
        const formData = new FormData();
        formData.append('audio_file', recordedBlob, 'recording.webm'); 
        formData.append('phrase', phraseTextToSend); 
        formData.append('language', language);
        formData.append('city_id', city.id);

        try {
            const response = await fetch(`${FLASK_API_URL}/api/upload`, {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ error: 'Unknown upload error' }));
                throw new Error(`Upload failed: ${errorData.error || response.statusText}`);
            }

            
            // If we fulfilled a request, mark it as fulfilled
            if (recordingRequestId) {
                 await fetch(`${FLASK_API_URL}/api/fulfill_request/${recordingRequestId}`, { method: 'POST' });
                 fetchRequests(city.id); // Refresh request list 
                 setRecordingRequestId(null);
                 setRecordingRequestText(null);
            }

            setRecordedBlob(null);
            setUploadStatus('success');
            setTimeout(() => setUploadStatus(null), 3000);
            fetchRecordings(city.id); // Refresh the recordings list
            if (onUploadSuccess) onUploadSuccess();
            
        } catch (error) {
            console.error('Error during upload:', error);
            setUploadStatus('error');
        } finally {
            setIsUploading(false);
        }
    };

    // --- Phrase Request Logic (Saves to phrase_requests table via Flask) ---
    const handlePhraseRequest = async (e) => {
        e.preventDefault();
        
        setIsRequesting(true);
        setPhraseRequestStatus(null);
        
        const requestData = {
            request_text: requestPhraseText.trim(),
            city_id: city.id,
            language: language 
        };

        try {
            const response = await fetch(`${FLASK_API_URL}/api/request_phrase`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestData),
            });

            if (!response.ok) {
                 const errorData = await response.json().catch(() => ({ error: 'Unknown request error' }));
                 throw new Error(`Request failed: ${errorData.error || response.statusText}`);
            }

            // Successful request
            setRequestPhraseText('');
            setPhraseRequestStatus('success');
            setTimeout(() => setPhraseRequestStatus(null), 3000);
            fetchRequests(city.id); // Refresh the request list immediately
            
        } catch (error) {
            console.error('Error during phrase request:', error);
            setPhraseRequestStatus('error');
        } finally {
            setIsRequesting(false);
        }
    };

    const recordedAudioURL = recordedBlob ? URL.createObjectURL(recordedBlob) : null;


    return (
        <div className="space-y-10">
            {/* Mic Error Modal (replaces alert) */}
            {uploadStatus === 'mic_error' && (
                <div className="fixed inset-0 z-[1001] flex items-center justify-center bg-black/50 backdrop-blur-sm p-4">
                    <div className="bg-white p-6 rounded-lg shadow-2xl max-w-sm text-center">
                        <AlertTriangle className="w-8 h-8 text-red-500 mx-auto mb-3"/>
                        <h3 className="text-lg font-bold text-gray-800 mb-2">Microphone Access Denied</h3>
                        <p className="text-sm text-gray-600">Please check your browser and system settings to ensure microphone access is permitted for this application.</p>
                        <button 
                            onClick={() => setUploadStatus(null)}
                            className="mt-4 px-4 py-2 bg-red-500 text-white rounded-lg hover:bg-red-600 transition-colors"
                        >
                            Close
                        </button>
                    </div>
                </div>
            )}

            <h2 className="text-3xl font-extrabold text-gray-800 flex items-center">
                <MapPin className="w-6 h-6 mr-3 text-red-600"/> 
                {city.name.trim()} - Contribution Hub
            </h2>
            <p className="text-gray-600">
                Listen to phrases recorded here or upload your own to capture the local accent.
            </p>

            {/* --- 1. UPLOAD FORM (Contribution Section) --- */}
            <div className="p-8 bg-indigo-50 border border-indigo-200 rounded-xl shadow-lg">
                <h3 className="text-2xl font-bold text-indigo-700 mb-6 flex items-center"><Mic className="w-6 h-6 mr-3"/> Record and Submit</h3>
                <form onSubmit={handleSubmit} className="space-y-6">
                    
                    {/* Display if fulfilling a request */}
                    {recordingRequestText && (
                        <div className="p-4 bg-green-100 border border-green-300 rounded-lg shadow-inner text-center">
                            <p className="font-semibold text-green-800">
                                Fulfilling Request: "{recordingRequestText}"
                            </p>
                            <button type="button" onClick={resetRecording} className="mt-2 text-sm text-red-500 hover:text-red-700">
                                Cancel Fulfillment
                            </button>
                        </div>
                    )}

                    {/* City-Specific Language Selection */}
                    <div>
                        <label className="block text-sm font-medium text-indigo-700 mb-1">Local Language Spoken (in {city.name.trim()}):</label>
                        <select 
                            value={language} 
                            onChange={(e) => setLanguage(e.target.value)}
                            className="w-full border border-indigo-300 rounded-lg p-3 focus:ring-indigo-500 focus:border-indigo-500 bg-white shadow-sm text-gray-900" 
                        >
                            {availableLanguages.map(lang => <option key={lang} value={lang}>{lang}</option>)}
                        </select>
                    </div>

                    {/* Fixed Phrase Selection (Hidden when fulfilling a request) */}
                    {!recordingRequestText && (
                        <div>
                            <label className="block text-sm font-medium text-indigo-700 mb-1 flex items-center"><BookOpen className="w-4 h-4 mr-2"/> English Phrase to Record:</label>
                            <select 
                                value={phraseId} 
                                onChange={(e) => setPhraseId(parseInt(e.target.value))}
                                required
                                className="w-full border border-indigo-300 rounded-lg p-3 focus:ring-indigo-500 focus:border-indigo-500 bg-white shadow-sm text-gray-900"
                            >
                                {phrasesList.length > 0 ? (
                                    phrasesList.map(p => (
                                        <option key={p.id} value={p.id}>
                                            {p.english_text}
                                        </option>
                                    ))
                                ) : (
                                    <option value="">Loading phrases...</option>
                                )}
                            </select>
                            <p className="mt-2 text-xs text-gray-500">
                                Select the English phrase, record how it sounds in the chosen local language, and upload.
                            </p>
                        </div>
                    )}
                    
                    {/* Recording Interface */}
                    <div className="pt-2 border-t pt-4">
                        {isRecording ? (
                            <button 
                                type="button" 
                                onClick={stopRecording} 
                                className="w-full bg-red-600 text-white py-3 px-4 rounded-full font-semibold hover:bg-red-700 flex items-center justify-center transition duration-150 shadow-lg shadow-red-300/50"
                            >
                                <StopCircle className="w-6 h-6 mr-3 animate-pulse"/> Stop Recording
                            </button>
                        ) : recordedBlob ? (
                            <div className="flex flex-col md:flex-row items-center justify-between bg-white p-4 rounded-xl border border-green-300 shadow-inner">
                                <div className="flex items-center w-full md:w-auto mb-3 md:mb-0">
                                    <p className="text-sm text-green-700 font-medium mr-4">Review Your Audio:</p>
                                    <audio ref={(el) => audioRefs.current['review'] = el} src={recordedAudioURL} className="flex-1 h-10 hidden" /> 
                                    <button 
                                        type="button" 
                                        onClick={() => togglePlayback('review')}
                                        className="p-2 border rounded-full text-indigo-600 hover:bg-indigo-100 transition"
                                    >
                                        {playingAudioId === 'review' ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
                                    </button>
                                </div>
                                <button 
                                    type="button" 
                                    onClick={resetRecording} 
                                    className="p-2 text-red-600 border border-red-300 rounded-full hover:bg-red-50 transition flex items-center text-sm"
                                    title="Re-record"
                                >
                                    <RotateCcw className="w-5 h-5 mr-1" /> Re-record
                                </button>
                            </div>
                        ) : (
                            <button 
                                type="button" 
                                onClick={startStandardRecording} 
                                className="w-full bg-green-600 text-white py-3 px-4 rounded-full font-semibold hover:bg-green-700 flex items-center justify-center transition duration-150 shadow-lg shadow-green-300/50"
                                disabled={!!recordingRequestText} // Disable if fulfilling a request via the board
                            >
                                <Mic className="w-6 h-6 mr-3"/> Start Recording
                            </button>
                        )}
                    </div>


                    {/* Submit Button and Status */}
                    <button 
                        type="submit" 
                        disabled={isUploading || !recordedBlob || !language || (!recordingRequestText && !phraseId)} // Check if phraseId is selected or if fulfilling a request
                        className="w-full bg-indigo-600 text-white py-3 px-4 rounded-full text-lg font-semibold hover:bg-indigo-700 disabled:bg-gray-400 disabled:cursor-not-allowed flex items-center justify-center transition duration-150 shadow-md"
                    >
                        {isUploading ? (
                            <>
                                <Loader className="w-5 h-5 mr-3 animate-spin"/> Uploading...
                            </>
                        ) : (
                            <><Send className="w-5 h-5 mr-3"/> Submit Contribution</>
                        )}
                    </button>

                    {uploadStatus === 'success' && (
                        <div className="p-3 bg-green-100 text-green-700 rounded-lg text-center font-medium shadow-inner">Upload Successful! List refreshing...</div>
                    )}
                    {uploadStatus === 'error' && (
                        <div className="p-3 bg-red-100 text-red-700 rounded-lg text-center font-medium shadow-inner flex items-center justify-center">
                            <AlertTriangle className="w-5 h-5 mr-2"/> Upload Failed. Check console for details.
                        </div>
                    )}
                </form>
            </div>
            
            {/* --- 2. PHRASE REQUEST SECTION --- */}
            <div className="p-6 bg-yellow-50 border border-yellow-200 rounded-xl shadow-md space-y-4">
                <h3 className="text-xl font-bold text-yellow-700 flex items-center"><MessageSquare className="w-5 h-5 mr-2"/> Request a New Phrase</h3>
                <p className="text-gray-600 text-sm">
                    Can't find the phrase you want to record? Request it here, and we'll add it to the list for other contributors!
                </p>
                <form onSubmit={handlePhraseRequest} className="space-y-4">
                 { /*language selection for requests phase */ } <div className="flex items-center space-x-3 justify-start"> 
                        <label htmlFor="request-language" className="block text-sm font-medium text-yellow-700 mb-1">Language:</label>
                       <select 
                          id ="request-language" //id for accesiitlity
                          value={language} 
                          onChange={(e) => setLanguage(e.target.value)}
                          required
                          className="w-29 border border-yellow-300 rounded-lg p-3 bg-white shadow-sm text-gray-900"
                        >
                       {availableLanguages.map(lang => <option key={`req-${lang}`} value={lang}>{lang}</option>)}
                       </select>
                    </div >
                    {/* inpuutt field */}
                     <div className="flex items-center space-x-3"> 
                    <div className="flex-grow"> 
                        <label htmlFor="request-phrase-text" className="sr-only">Type the new English phrase</label>
                    <input
                        id= "request-phrase-text"
                        type="text"
                        value={requestPhraseText}
                        onChange={(e) => setRequestPhraseText(e.target.value)}
                        required
                        className="w-full border border-yellow-300 rounded-lg p-3 focus:ring-yellow-500 focus:border-yellow-500 shadow-sm text-gray-900" 
                            
                        placeholder="Type the new English phrase.."
                    />
                     </div>
                      <div className="flex-none">
                    <button
                        type="submit"
                       disabled={isRequesting || !language} /* Added check for language selection */
                       className="flex-none bg-yellow-600 text-white py-3 px-4 rounded-lg font-semibold hover:bg-yellow-700 disabled:bg-gray-400 flex items-center justify-center transition duration-150 shadow-md whitespace-nowrap"
            >
                        {isRequesting ? (
                            <><Loader className="w-5 h-5 mr-2 animate-spin"/> Submitting...</>
                        ) : (
                            <><PlusCircle className="w-5 h-5 mr-2"/> Request</>
                        )}
                    </button>
                    </div>
                    </div>
                </form>
                {phraseRequestStatus === 'success' && (
                    <div className="p-2 text-yellow-800 text-center text-sm font-medium">Request noted! Thank you for your contribution idea.</div>
                )}
                {phraseRequestStatus === 'error' && (
                    <div className="p-2 text-red-700 text-center text-sm font-medium">Request failed to submit. Check console for details.</div>
                )}
            </div>
            
            {/* --- 3. COMMUNITY REQUEST BOARD --- */}
            <div className="p-8 bg-white border border-gray-200 rounded-xl shadow-lg">
                <h3 className="text-2xl font-bold text-gray-800 mb-6 flex items-center">
                    <UserCheck className="w-6 h-6 mr-3 text-blue-600"/> Community Request Board ({pendingRequests.length})
                </h3>

                {isRequestsLoading ? (
                    <div className="text-center py-4">
                        <Loader className="w-6 h-6 text-gray-400 animate-spin mx-auto"/>
                        <p className="text-gray-500 mt-2 text-sm">Fetching pending requests...</p>
                    </div>
                ) : pendingRequests.length === 0 ? (
                    <p className="text-gray-500 text-center py-4 border-t pt-4">No pending phrase requests for {city.name.trim()}.</p>
                ) : (
                    <div className="border-t pt-4 space-y-3">
                        <p className="text-sm text-gray-600 font-medium">Click "Record" next to a phrase to contribute your local accent!</p>
                        <ul className="space-y-3">
                            {pendingRequests.map(req => (
                                <li key={req.id} className={`p-4 rounded-lg flex flex-col md:flex-row items-center justify-between shadow-sm transition duration-150 
                                    ${recordingRequestId === req.id ? 'bg-indigo-100 border-indigo-500' : 'bg-blue-50 border-blue-200'}`}
                                >
                                    <span className="flex-none bg-indigo-600 text-white text-xs font-bold px-3 py-1 rounded-full whitespace-nowrap shadow-md">
                                {req.language ? req.language.toUpperCase() : 'UNKNOWN'} 
                                  </span>
                                    <div className="flex-1 min-w-0 flex items-center">
                                        <MessageSquare className="w-4 h-4 mr-2 text-blue-600"/> 
                                        <p className="font-semibold text-blue-800 truncate">{req.request_text}</p>
                                    </div>
                                    
                                    {/* Conditional Recording Controls for this specific item */}
                                    {recordingRequestId === req.id ? (
                                        <div className="mt-3 md:mt-0 flex flex-col items-center space-y-2 w-full md:w-auto">
                                            {isRecording ? (
                                                <button 
                                                    type="button" 
                                                    onClick={stopRecording} 
                                                    className="w-full bg-red-600 text-white py-2 px-4 rounded-full font-semibold hover:bg-red-700 flex items-center justify-center transition"
                                                >


                                                    <StopCircle className="w-5 h-5 mr-2 animate-pulse"/> Stop Recording
                                                </button>
                                            ) : recordedBlob ? (
                                                <div className="flex items-center space-x-2">
                                                    <audio ref={(el) => audioRefs.current[`req-review-${req.id}`] = el} src={recordedAudioURL} className="hidden"/>
                                                    <button 
                                                        type="button"
                                                        onClick={() => togglePlayback(`req-review-${req.id}`)}
                                                        className="p-1 border rounded-full text-indigo-600 hover:bg-indigo-100 transition"
                                                    >
                                                        {playingAudioId === `req-review-${req.id}` ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
                                                    </button>
                                                    
                                                    <button 
                                                        type="button" 
                                                        onClick={resetRecording} 
                                                        className="p-1 text-red-600 border border-red-300 rounded-full hover:bg-red-50 transition"
                                                        title="Re-record"
                                                    >
                                                        <RotateCcw className="w-4 h-4" />
                                                    </button>
                                                    
                                                    <button
                                                        type="button" 
                                                        onClick={handleSubmit} 
                                                        disabled={isUploading || !language}
                                                        className="w-full bg-indigo-600 text-white py-2 px-4 rounded-full text-sm font-semibold hover:bg-indigo-700 disabled:bg-gray-400 flex items-center justify-center transition"
                                                    >
                                                        {isUploading ? <Loader className="w-4 h-4 mr-2 animate-spin"/> : <Send className="w-4 h-4 mr-2"/>}
                                                        Submit Recording
                                                    </button>
                                                </div>
                                            ) : null}
                                        </div>
                                    ) : (
                                        // Initial 'Record' button
                                        <button
                                            type="button"
                                            onClick={() => startRecordingForRequest(req.id, req.request_text)}
                                            disabled={isRecording || !!recordingRequestText} // Disable if recording anywhere else
                                            className="bg-green-500 text-white py-2 px-3 rounded-full text-sm font-semibold hover:bg-green-600 disabled:bg-gray-400 transition flex items-center"
                                        >
                                            <Mic className="w-4 h-4 mr-1"/> Record
                                        </button>
                                    )}
                                </li>
                            ))}
                        </ul>
                    </div>
                )}
            </div>

            {/* --- 4. RECORDINGS LIST (Listen Section) --- */}
            <div className="p-8 bg-white border border-gray-200 rounded-xl shadow-lg">
                <h3 className="text-2xl font-bold text-gray-800 mb-6 flex items-center"><Play className="w-6 h-6 mr-3"/> Listen to Accents ({recordings.length})</h3>
                
                {fetchError && (
                    <div className="p-4 bg-red-100 text-red-700 rounded-lg flex items-center mb-4">
                        <AlertTriangle className="w-5 h-5 mr-2"/> {fetchError}
                    </div>
                )}
                
                {isRecordingsLoading ? (
                    <div className="text-center py-8">
                        <Loader className="w-8 h-8 text-gray-400 animate-spin mx-auto"/>
                        <p className="text-gray-500 mt-3">Fetching audio clips and organizing by language...</p>
                    </div>
                ) : recordings.length === 0 ? (
                    <p className="text-gray-500 text-center py-6">No recordings found for {city.name.trim()}. Be the first to contribute!</p>
                ) : (
                    <div className="space-y-6">
                        {Object.entries(groupedRecordings).map(([language, list]) => (
                            <div key={language} className="border-t pt-4">
                                <h4 className="text-xl font-semibold text-gray-700 mb-3">{language} ({list.length} clips)</h4>
                                <ul className="space-y-3">
                                    {list.map(rec => (
                                        <li key={rec.id} className="p-4 border border-gray-100 rounded-xl flex flex-col md:flex-row justify-between items-start md:items-center bg-gray-50 hover:bg-gray-100 transition duration-100 shadow-sm">
                                            <div className="flex-1 min-w-0 mb-2 md:mb-0">
                                                <p className="font-semibold text-gray-800">Phrase: {rec.english_text}</p>
                                                <p className="text-xs text-gray-500 mt-1">Uploaded: {new Date(rec.created_at).toLocaleDateString()}</p>
                                            </div>
                                            <div className="ml-0 md:ml-4 flex items-center justify-end w-full md:w-auto">
                                                {/* Custom Playback for Listening (No Download) */}
                                                <audio ref={(el) => audioRefs.current[rec.id] = el} src={rec.audio_url} className="hidden" />
                                                <button
                                                    type="button"
                                                    onClick={() => togglePlayback(rec.id)}
                                                    className="bg-blue-500 text-white p-2 rounded-full hover:bg-blue-600 transition duration-150 shadow-md flex items-center justify-center"
                                                >
                                                    {playingAudioId === rec.id ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
                                                </button>
                                            </div>
                                        </li>
                                    ))}
                                </ul>
                            </div>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
};
export default CityPanel;


