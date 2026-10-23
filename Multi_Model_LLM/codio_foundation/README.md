Codio Foundation encapsulate the following interesting learnings:
- fetch an image from file
- using the path library to work with files
- check if the path to a file holds really a file (is_file)
- check if the image is really an image
- encode an image as ascii using base64.b64encode
- derive the mimetype using: mimetypes= mimetypes.guess_type(path.name)[0] or "image/png"
- use openAI to send string of encoded image as ascii string to the model. Model will decode ascii back to bytes and interpret the bytes using the mimetypes that was sent along with the string
- Send image and text (instruction) to modal to get the model answering the questions from the instructions. like, what's in the picture
- compact the text response of the model 