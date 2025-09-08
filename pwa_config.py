import streamlit as st
import json
from pathlib import Path

# Progressive Web App Configuration
class PWAConfig:
    def __init__(self):
        self.app_name = "Advanced Multi-Hop RAG Agent"
        self.short_name = "RAG Agent"
        self.description = "AI-powered financial and sports analysis with real-time data"
        self.theme_color = "#3b82f6"
        self.background_color = "#0f172a"
        
    def generate_manifest(self) -> dict:
        """Generate PWA manifest"""
        return {
            "name": self.app_name,
            "short_name": self.short_name,
            "description": self.description,
            "start_url": "/",
            "display": "standalone",
            "background_color": self.background_color,
            "theme_color": self.theme_color,
            "orientation": "portrait-primary",
            "icons": [
                {
                    "src": "/static/icon-192.png",
                    "sizes": "192x192",
                    "type": "image/png",
                    "purpose": "any maskable"
                },
                {
                    "src": "/static/icon-512.png", 
                    "sizes": "512x512",
                    "type": "image/png",
                    "purpose": "any maskable"
                }
            ],
            "categories": ["finance", "business", "productivity"],
            "lang": "en",
            "dir": "ltr"
        }
    
    def generate_service_worker(self) -> str:
        """Generate service worker for offline functionality"""
        return """
const CACHE_NAME = 'rag-agent-v1';
const urlsToCache = [
    '/',
    '/static/css/main.css',
    '/static/js/main.js',
    '/static/icon-192.png',
    '/static/icon-512.png'
];

// Install event
self.addEventListener('install', event => {
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then(cache => {
                return cache.addAll(urlsToCache);
            })
    );
});

// Fetch event
self.addEventListener('fetch', event => {
    event.respondWith(
        caches.match(event.request)
            .then(response => {
                // Return cached version or fetch from network
                return response || fetch(event.request);
            }
        )
    );
});

// Activate event
self.addEventListener('activate', event => {
    event.waitUntil(
        caches.keys().then(cacheNames => {
            return Promise.all(
                cacheNames.map(cacheName => {
                    if (cacheName !== CACHE_NAME) {
                        return caches.delete(cacheName);
                    }
                })
            );
        })
    );
});

// Push notification event
self.addEventListener('push', event => {
    const options = {
        body: event.data ? event.data.text() : 'New analysis available!',
        icon: '/static/icon-192.png',
        badge: '/static/icon-192.png',
        vibrate: [100, 50, 100],
        data: {
            dateOfArrival: Date.now(),
            primaryKey: 1
        },
        actions: [
            {
                action: 'explore',
                title: 'View Analysis',
                icon: '/static/icon-192.png'
            },
            {
                action: 'close',
                title: 'Close',
                icon: '/static/icon-192.png'
            }
        ]
    };
    
    event.waitUntil(
        self.registration.showNotification('RAG Agent', options)
    );
});
"""

    def inject_pwa_html(self) -> str:
        """Generate HTML to inject PWA features"""
        manifest_json = json.dumps(self.generate_manifest(), indent=2)
        
        return f"""
<script>
// PWA Installation
let deferredPrompt;
let installButton = null;

window.addEventListener('beforeinstallprompt', (e) => {{
    e.preventDefault();
    deferredPrompt = e;
    showInstallButton();
}});

function showInstallButton() {{
    if (!installButton) {{
        installButton = document.createElement('button');
        installButton.innerHTML = '📱 Install App';
        installButton.style.cssText = `
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
            color: white;
            border: none;
            padding: 12px 20px;
            border-radius: 25px;
            font-weight: 500;
            cursor: pointer;
            box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
            z-index: 1000;
            font-size: 14px;
            transition: all 0.3s ease;
        `;
        
        installButton.addEventListener('mouseover', () => {{
            installButton.style.transform = 'translateY(-2px)';
            installButton.style.boxShadow = '0 8px 25px rgba(59, 130, 246, 0.4)';
        }});
        
        installButton.addEventListener('mouseout', () => {{
            installButton.style.transform = 'translateY(0)';
            installButton.style.boxShadow = '0 4px 15px rgba(59, 130, 246, 0.3)';
        }});
        
        installButton.addEventListener('click', installApp);
        document.body.appendChild(installButton);
    }}
}}

async function installApp() {{
    if (deferredPrompt) {{
        deferredPrompt.prompt();
        const {{ outcome }} = await deferredPrompt.userChoice;
        
        if (outcome === 'accepted') {{
            console.log('User accepted the install prompt');
            hideInstallButton();
        }}
        
        deferredPrompt = null;
    }}
}}

function hideInstallButton() {{
    if (installButton) {{
        installButton.remove();
        installButton = null;
    }}
}}

// Service Worker Registration
if ('serviceWorker' in navigator) {{
    window.addEventListener('load', () => {{
        navigator.serviceWorker.register('/sw.js')
            .then(registration => {{
                console.log('SW registered: ', registration);
            }})
            .catch(registrationError => {{
                console.log('SW registration failed: ', registrationError);
            }});
    }});
}}

// Push Notifications
async function requestNotificationPermission() {{
    if ('Notification' in window) {{
        const permission = await Notification.requestPermission();
        return permission === 'granted';
    }}
    return false;
}}

// Touch gestures for mobile
let touchStartX = 0;
let touchStartY = 0;

document.addEventListener('touchstart', e => {{
    touchStartX = e.changedTouches[0].screenX;
    touchStartY = e.changedTouches[0].screenY;
}}, false);

document.addEventListener('touchend', e => {{
    const touchEndX = e.changedTouches[0].screenX;
    const touchEndY = e.changedTouches[0].screenY;
    
    const deltaX = touchEndX - touchStartX;
    const deltaY = touchEndY - touchStartY;
    
    // Swipe gestures
    if (Math.abs(deltaX) > Math.abs(deltaY)) {{
        if (deltaX > 50) {{
            // Swipe right - could trigger sidebar
            console.log('Swipe right detected');
        }} else if (deltaX < -50) {{
            // Swipe left - could hide sidebar
            console.log('Swipe left detected');
        }}
    }}
}}, false);

// Offline detection
window.addEventListener('online', () => {{
    console.log('Back online');
    showNotification('Connection restored', 'success');
}});

window.addEventListener('offline', () => {{
    console.log('Gone offline');
    showNotification('Working offline', 'warning');
}});

function showNotification(message, type = 'info') {{
    const notification = document.createElement('div');
    notification.innerHTML = message;
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 12px 20px;
        border-radius: 8px;
        color: white;
        font-weight: 500;
        z-index: 1001;
        animation: slideIn 0.3s ease;
        background: ${{type === 'success' ? '#10b981' : type === 'warning' ? '#f59e0b' : '#3b82f6'}};
    `;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {{
        notification.remove();
    }}, 3000);
}}

// Add CSS animation
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {{
        from {{ transform: translateX(100%); opacity: 0; }}
        to {{ transform: translateX(0); opacity: 1; }}
    }}
    
    @media (max-width: 768px) {{
        .stSidebar {{
            transform: translateX(-100%);
            transition: transform 0.3s ease;
        }}
        
        .stSidebar.mobile-open {{
            transform: translateX(0);
        }}
        
        .main .block-container {{
            padding-left: 1rem;
            padding-right: 1rem;
        }}
    }}
`;
document.head.appendChild(style);
</script>

<link rel="manifest" href="data:application/json;base64,{json.dumps(manifest_json).encode().hex()}">
<meta name="theme-color" content="{self.theme_color}">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="{self.short_name}">
<meta name="mobile-web-app-capable" content="yes">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
"""

# Mobile Optimization Functions
def optimize_for_mobile():
    """Apply mobile-specific optimizations"""
    
    # Detect mobile device
    mobile_css = """
    <style>
    @media (max-width: 768px) {
        /* Mobile-first responsive design */
        .stApp {
            padding: 0.5rem;
        }
        
        .main-header {
            padding: 1.5rem 1rem;
            font-size: 1.2rem;
        }
        
        .main-header h1 {
            font-size: 1.8rem !important;
        }
        
        .main-header p {
            font-size: 1rem !important;
        }
        
        /* Touch-friendly buttons */
        .stButton > button {
            min-height: 44px;
            font-size: 16px;
            padding: 12px 20px;
        }
        
        /* Larger input fields */
        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea {
            font-size: 16px;
            min-height: 44px;
        }
        
        /* Optimized metrics */
        .metric-card {
            padding: 0.75rem;
            margin: 0.5rem 0;
        }
        
        /* Sidebar optimization */
        .stSidebar {
            width: 280px;
        }
        
        /* Chart optimization */
        .js-plotly-plot {
            width: 100% !important;
        }
        
        /* Hide desktop-only elements */
        .desktop-only {
            display: none !important;
        }
        
        /* Show mobile-only elements */
        .mobile-only {
            display: block !important;
        }
        
        /* Swipe indicators */
        .swipe-indicator {
            position: fixed;
            bottom: 10px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(0,0,0,0.7);
            color: white;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 12px;
            z-index: 1000;
        }
    }
    
    @media (min-width: 769px) {
        .mobile-only {
            display: none !important;
        }
        
        .desktop-only {
            display: block !important;
        }
    }
    
    /* Touch feedback */
    .stButton > button:active,
    .stSelectbox:active {
        transform: scale(0.98);
        transition: transform 0.1s ease;
    }
    
    /* Loading states */
    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid #f3f3f3;
        border-top: 3px solid #3b82f6;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    </style>
    """
    
    st.markdown(mobile_css, unsafe_allow_html=True)

def add_mobile_navigation():
    """Add mobile-specific navigation"""
    
    # Mobile menu toggle
    if st.button("☰", key="mobile_menu", help="Toggle menu"):
        st.session_state.mobile_menu_open = not st.session_state.get('mobile_menu_open', False)
    
    # Swipe indicators
    st.markdown("""
    <div class="mobile-only swipe-indicator">
        ← Swipe for menu | Swipe for actions →
    </div>
    """, unsafe_allow_html=True)

def create_mobile_shortcuts():
    """Create mobile-friendly shortcuts"""
    
    shortcuts_html = """
    <div class="mobile-only" style="
        position: fixed;
        bottom: 60px;
        right: 20px;
        display: flex;
        flex-direction: column;
        gap: 10px;
        z-index: 1000;
    ">
        <button onclick="scrollToTop()" style="
            width: 50px;
            height: 50px;
            border-radius: 50%;
            background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
            color: white;
            border: none;
            font-size: 20px;
            cursor: pointer;
            box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
        ">↑</button>
        
        <button onclick="toggleDarkMode()" style="
            width: 50px;
            height: 50px;
            border-radius: 50%;
            background: linear-gradient(135deg, #8b5cf6 0%, #ec4899 100%);
            color: white;
            border: none;
            font-size: 20px;
            cursor: pointer;
            box-shadow: 0 4px 15px rgba(139, 92, 246, 0.3);
        ">🌙</button>
    </div>
    
    <script>
    function scrollToTop() {
        window.scrollTo({top: 0, behavior: 'smooth'});
    }
    
    function toggleDarkMode() {
        // Trigger dark mode toggle
        const darkModeBtn = document.querySelector('button[title*="Dark"]') || 
                           document.querySelector('button[title*="Light"]');
        if (darkModeBtn) darkModeBtn.click();
    }
    </script>
    """
    
    st.markdown(shortcuts_html, unsafe_allow_html=True)

# Export functions
__all__ = [
    'PWAConfig',
    'optimize_for_mobile', 
    'add_mobile_navigation',
    'create_mobile_shortcuts'
]