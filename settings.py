# أسماء الكلاسات بالترتيب الصحيح للموديل

CLASS_NAMES = [
    "Corn_(maize)___Common_rust_",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___healthy",
]

# أسماء تظهر للمستخدم بشكل أحسن
DISPLAY_NAMES = {
    "Corn_(maize)___Common_rust_": "Corn — Common Rust",
    "Pepper,_bell___Bacterial_spot": "Pepper — Bacterial Spot",
    "Pepper,_bell___healthy": "Pepper — Healthy",
    "Potato___Early_blight": "Potato — Early Blight",
    "Potato___Late_blight": "Potato — Late Blight",
    "Potato___healthy": "Potato — Healthy",
    "Tomato___Early_blight": "Tomato — Early Blight",
    "Tomato___Late_blight": "Tomato — Late Blight",
    "Tomato___Leaf_Mold": "Tomato — Leaf Mold",
    "Tomato___healthy": "Tomato — Healthy",
}

# معلومات الأمراض
DISEASE_INFO = {
    "Corn_(maize)___Common_rust_": {
        "description": "Fungal disease causing rust-colored spots on corn leaves.",
        "next_step": "Apply suitable fungicide and remove infected leaves."
    },

    "Pepper,_bell___Bacterial_spot": {
        "description": "Bacterial disease causing dark spots on pepper leaves.",
        "next_step": "Avoid overhead watering and use copper-based sprays."
    },

    "Pepper,_bell___healthy": {
        "description": "The pepper plant appears healthy.",
        "next_step": "No action required."
    },

    "Potato___Early_blight": {
        "description": "Fungal disease causing brown concentric spots.",
        "next_step": "Remove infected leaves and apply fungicide."
    },

    "Potato___Late_blight": {
        "description": "Serious fungal disease causing dark lesions.",
        "next_step": "Use fungicide immediately and isolate infected plants."
    },

    "Potato___healthy": {
        "description": "The potato plant appears healthy.",
        "next_step": "No action required."
    },

    "Tomato___Early_blight": {
        "description": "Fungal disease causing target-like spots.",
        "next_step": "Remove infected leaves and improve air circulation."
    },

    "Tomato___Late_blight": {
        "description": "Aggressive disease causing dark wet lesions.",
        "next_step": "Apply fungicide and remove infected plants."
    },

    "Tomato___Leaf_Mold": {
        "description": "Fungal disease causing yellow patches and mold.",
        "next_step": "Reduce humidity and apply fungicide."
    },

    "Tomato___healthy": {
        "description": "The tomato plant appears healthy.",
        "next_step": "No action required."
    },
}