import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split, cross_val_predict, KFold
from sklearn.metrics import mean_squared_error
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, ConversationHandler, CallbackContext

# Scores des matchs passés (entrez vos données ici)
scores = np.array([[1, 2], [2, 3], [3, 1], [2, 2], [1, 0]])
but_marques = scores[:, 0]  # Buts marqués par l'équipe
but_encaisses = scores[:, 1]  # Buts encaissés par l'équipe

# Créez une fonction pour construire le modèle avec des hyperparamètres ajustables
def build_model(num_layers=2, layer_size=64):
    modele = keras.Sequential()
    modele.add(keras.layers.Dense(layer_size, activation='relu', input_shape=(1,)))
    
    for _ in range(num_layers - 1):
        modele.add(keras.layers.Dense(layer_size, activation='relu'))
    
    modele.add(keras.layers.Dense(1))  # Couche de sortie
    modele.compile(optimizer='adam', loss='mean_squared_error')
    return modele

# Créez un modèle KerasRegressor
modele = KerasRegressor(build_fn=build_model, epochs=100, batch_size=2, verbose=0)

# Divisez les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(but_marques, but_encaisses, test_size=0.2, random_state=42)

# Définissez le nombre de plis (folds) pour la validation croisée par défaut
kf = KFold(shuffle=True, random_state=42)

# Créez une fonction pour effectuer la validation croisée et obtenir les prédictions
def perform_cross_validation(X, y, model):
    y_pred = cross_val_predict(model, X, y, cv=kf)
    return y_pred

# Appelez la fonction perform_cross_validation pour obtenir les prédictions
y_pred = perform_cross_validation(X_train, y_train, modele)

# Définissez les étapes de la conversation
SCORE, FINISH = range(2)

# Fonction pour gérer la commande /start
def start(update: Update, context: CallbackContext) -> int:
    update.message.reply_text("Bienvenue ! Veuillez entrer les scores des matchs passés.")
    return SCORE

# Fonction pour gérer l'entrée des scores et afficher les prédictions
def score(update: Update, context: CallbackContext) -> int:
    user = update.message.from_user
    text = update.message.text
    
    # Utilisez les prédictions y_pred ici
    prediction_message = "Prédictions pour les prochains matchs :\n"
    for i, pred in enumerate(y_pred):
        prediction_message += f"Match {i + 1}: {pred} buts\n"
    
    # Envoyez les prédictions au chat
    update.message.reply_text(f"Scores enregistrés : {text}")
    update.message.reply_text(prediction_message)
    
    return FINISH

# Fonction pour gérer la fin de la conversation
def finish(update: Update, context: CallbackContext) -> int:
    update.message.reply_text("Merci d'avoir entré les scores des matchs passés !")
    return ConversationHandler.END

def main():
    updater = Updater("6364693971:AAG7L_L4sUiih_En7StmW5U6LdqYlj_2SGU", use_context=True)  # Nouveau token d'API
    
    dp = updater.dispatcher
    
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            SCORE: [MessageHandler(Filters.text & ~Filters.command, score)],
            FINISH: [MessageHandler(Filters.text & ~Filters.command, finish)],
        },
        fallbacks=[],
    )
    
    dp.add_handler(conv_handler)
    
    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()
