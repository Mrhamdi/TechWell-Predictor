import pandas as pd                     # gérer données
from fastapi import FastAPI             # créer API
from fastapi.middleware.cors import CORSMiddleware  # gérer CORS
from pydantic import BaseModel         # validation données
from sklearn.model_selection import train_test_split  # séparer dataset
from sklearn.preprocessing import LabelEncoder        # encoder labels
from sklearn.ensemble import RandomForestClassifier   # modèle arbre
import uvicorn                          # serveur API

# ============================================================
# 1) Charger et préparer la DATASET
# ============================================================

df = pd.read_csv("survey.csv")          # lire fichier

df = df[[                               
    "Age",                              # âge utilisateur
    "Gender",                           # genre utilisateur
    "family_history",                   # historique famille
    "work_interfere",                   # travail interfère
    "benefits",                         # avantages employés
    "care_options",                      # options soins
    "seek_help",                         # cherche aide
    "mental_health_consequence"          # conséquence mentale
]]                                      

df = df[df["Age"].between(15, 80)]      # filtrer âge

label_encoders = {}                      # dictionnaire encodeurs
for col in df.columns:                   # pour chaque colonne
    if df[col].dtype == "object":       # si texte
        le = LabelEncoder()             # créer encodeur
        df[col] = le.fit_transform(df[col].astype(str))  # encoder colonne
        label_encoders[col] = le        # sauvegarder encodeur

X = df.drop("seek_help", axis=1)        # features dataset
y = df["seek_help"]                     # target dataset

X_train, X_test, y_train, y_test = train_test_split(  # split dataset
    X, y, test_size=0.2, random_state=42
)

# ============================================================
# 2) Entraîner le modèle
# ============================================================

model = RandomForestClassifier()         # créer modèle
model.fit(X_train, y_train)              # entraîner modèle

# ============================================================
# 3) API FASTAPI
# ============================================================

app = FastAPI(title="Mental Health Prediction API")  # init API

# ======== Ajouter CORS Middleware ========
app.add_middleware(
    CORSMiddleware,                      # activer CORS
    allow_origins=["http://127.0.0.1:5500", "http://localhost:5500"],  # front-end local
    allow_credentials=True,              # cookies autorisés
    allow_methods=["*"],                 # toutes méthodes
    allow_headers=["*"],                 # tous headers
)

# Schéma d’entrée
class UserInput(BaseModel):              # validation entrée
    Age: int
    Gender: str
    family_history: str
    work_interfere: str
    benefits: str
    care_options: str
    mental_health_consequence: str


@app.post("/predict")                     # route POST
def predict(data: UserInput):            # fonction prédiction
    input_df = pd.DataFrame([data.dict()])  # convertir DataFrame

    for col in input_df.columns:         # pour chaque colonne
        if col in label_encoders:       # si encodage existe
            le = label_encoders[col]    # récupérer encodeur
            input_df[col] = le.transform(input_df[col].astype(str))  # encoder valeur

    prediction = model.predict(input_df)[0]  # prédire résultat
    result = "YES (besoin d’aide psychologique)" if prediction == 1 else "NO (pas urgent)"  # texte résultat

    return {"prediction": int(prediction), "result": result}  # renvoyer JSON

# ============================================================
# 4) Lancer le serveur
# ============================================================
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)  # démarrer API
