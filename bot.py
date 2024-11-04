import telebot
from image_processing import LaserPatternDetector

BOT_TOKEN = '7725821106:AAG6P6iSYWKQ2vs38BVzlSAHaEokZk-KdI4'

bot = telebot.TeleBot(BOT_TOKEN)

@bot.message_handler(commands=['start'])
def test(message):
    bot.reply_to(message, 'Welcome to the bot! Type /test to test the bot.')

@bot.message_handler(commands=['test'])
def test(message):
    bot.reply_to(message, 'Hi')

@bot.message_handler(commands=['laser'])
def laser(message):
    bot.send_message(message.chat.id, 'Shooting laser pattern detection...')
    # Define a pattern function
    def pattern_function():
        return [0, 1, 0, 1, 1, 0, 1, 0, 0]

    laser_pattern_detector = LaserPatternDetector(camera_index=0, delay=0.2, pattern_function=pattern_function, n_frames=9)
    laser_pattern_detector.capture_frames()
    laser_spot = laser_pattern_detector.detect_laser_pattern()

    bot.reply_to(message, 'Laser pattern found at {}'.format(laser_spot))

bot.infinity_polling()
