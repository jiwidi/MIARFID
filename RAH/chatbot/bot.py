import logging

from telegram import ReplyKeyboardMarkup, ReplyKeyboardRemove, Update
from telegram.ext import (
    Updater,
    CommandHandler,
    MessageHandler,
    Filters,
    ConversationHandler,
    CallbackContext,
)

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

logger = logging.getLogger(__name__)

PEDIDO, TOPPINGS, EXTRA_TOPPINGS, LOCATION, END = range(5)


def start(update: Update, context: CallbackContext) -> int:
    reply_keyboard = [["Margarita", "Napoli", "4-Cheeses", "CUSTOM"]]
    update.message.reply_text(
        "Hi! My name is RAH-MIARFID Bot. You can order your pizza here. "
        "Send /cancel to stop talking to me.\n\n"
        "What type of pizza would you like to order??",
        reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True),
    )

    return PEDIDO


def custom(update: Update, context: CallbackContext) -> int:
    reply_keyboard = [["Blue cheese", "Olives", "Bacon", "Pineapple"]]
    update.message.reply_text(
        "I see! Lets start with what toppings would you like? Type /skip to finish the pizza customization",
        reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True),
    )

    return TOPPINGS


def toppings(update: Update, context: CallbackContext) -> int:
    reply_keyboard = [["Blue cheese", "Olives", "Bacon", "Pineapple"]]
    update.message.reply_text(
        "I see! Would you like to add any extra toppings? Type /skip to finish the pizza customization",
        reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True),
    )

    return TOPPINGS


def extra_toppings(update: Update, context: CallbackContext) -> int:
    reply_keyboard = [["Blue cheese", "Olives", "Bacon", "Pineapple"]]
    update.message.reply_text(
        "Good choice! More extra toppings? Type /skip to finish the pizza customization",
        reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True),
    )

    return EXTRA_TOPPINGS


def location(update: Update, context: CallbackContext) -> int:
    update.message.reply_text(
        "Gorgeous! Now, send me your delivery location please, "
        "or send /skip if you will come pick it up at store."
    )

    return LOCATION


def skip_toppings(update: Update, context: CallbackContext) -> int:
    update.message.reply_text(
        "I bet you that pizza will taste great! Now, send me your location please, "
        "or send /skip if you will come pick it up at store."
    )

    return LOCATION


def end_store(update: Update, context: CallbackContext) -> int:
    user = update.message.from_user
    update.message.reply_text(
        "Thank you! Your order number is 46290 and should be ready to pick in 20~30 minutes.I hope you order from us again in the future."
    )

    return ConversationHandler.END


def end_delivery(update: Update, context: CallbackContext) -> int:
    user = update.message.from_user
    update.message.reply_text(
        "Thank you! Your order number is 46290 and should arrive in 20~30 minutes. I hope you order from us again in the future."
    )

    return ConversationHandler.END


def cancel(update: Update, context: CallbackContext) -> int:
    user = update.message.from_user
    logger.info("User %s canceled the conversation.", user.first_name)
    update.message.reply_text(
        "Bye! I hope you order from us again in the future.",
        reply_markup=ReplyKeyboardRemove(),
    )

    return ConversationHandler.END


def main() -> None:
    # Create the Updater and pass it your bot's token.
    # Make sure to set use_context=True to use the new context based callbacks
    # Post version 12 this will no longer be necessary
    updater = Updater(
        "1409406611:AAGRIDmiLxQkRhStcw8XojBc29OZtCaoKM4", use_context=True
    )

    # Get the dispatcher to register handlers
    dispatcher = updater.dispatcher

    # Add conversation handler with the states PEDIDO, TOPPINGS, LOCATION and BIO
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            PEDIDO: [
                MessageHandler(
                    Filters.regex("^(Margarita|Napoli|4-Cheeses)$"), toppings,
                ),
                MessageHandler(Filters.regex("^(CUSTOM)$"), custom),
            ],
            TOPPINGS: [
                MessageHandler(
                    Filters.regex("^(Blue cheese|Olives|Bacon|Pineapple)$"),
                    extra_toppings,
                ),
                CommandHandler("skip", location),
            ],
            EXTRA_TOPPINGS: [
                MessageHandler(
                    Filters.regex("^(Blue cheese|Olives|Bacon|Pineapple)$"),
                    extra_toppings,
                ),
                CommandHandler("skip", location),
            ],
            LOCATION: [
                MessageHandler(Filters.location, end_delivery),
                CommandHandler("skip", end_store),
                MessageHandler(Filters.text, end_delivery),
            ],
            END: [MessageHandler(Filters.text & ~Filters.command, end_store)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )

    dispatcher.add_handler(conv_handler)

    # Start the Bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()


if __name__ == "__main__":
    main()
