import emoji
import unicodedata
import re

def contains_numbers(input_string):
	#If the token contains numbers and no hyphen we would like to detect that.
	return bool(re.search(r'\d', input_string)) and ('-' not in input_string)

def is_all_numbers_and_hyphens(input_string):
	return all([(c.isdigit() or c == '-') for c in input_string])

def is_all_punctuation(input_string):
	categories = [unicodedata.category(c).startswith('P') for c in input_string.strip()]
	return all(categories)

def is_emoji(s):
	return emoji.is_emoji(s)

def is_all_emojis(input_string):
	return all((is_emoji(c) or c=='\u200d') for c in input_string.strip())

def has_some_emoji(input_string):
	return any(is_emoji(c) for c in input_string.strip())

def is_all_punctuation_or_spaces_or_control(input_string):
	categories = [(unicodedata.category(c).startswith('P') 
		or unicodedata.category(c).startswith('Z') 
		or unicodedata.category(c).startswith('C')) for c in input_string.strip()]
	return all(categories)

def is_all_emoji_punctuation_or_spaces_or_control(input_string):
	categories = [(is_emoji(c)
		or unicodedata.category(c).startswith('P') 
		or unicodedata.category(c).startswith('Z') 
		or unicodedata.category(c).startswith('C')) for c in input_string.strip()]
	return all(categories)

def is_all_alpha_num_hyphen(input_string):
	return all([(c.isalnum() or c == '-' or c == "'") for c in input_string])

def is_all_alpha_num_hyphen_emoji(input_string):
	return all([(c.isalnum() or c == '-' or c == "'" or is_emoji(c)) for c in input_string])

def is_url(input_string):
	# return '.com' in input_string or 'http' in input_string
	url_regex = r"http\S+"
	return bool(re.search(re.compile(url_regex), input_string))

def is_phone_number(input_string):
	phone_number_regex=r"(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})"
	phone_number_regex_comp = re.compile(phone_number_regex)
	return bool(re.search(phone_number_regex_comp, input_string))

def is_hashtag(input_string):
	return input_string.startswith('#')
