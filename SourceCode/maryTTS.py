#############################################################################################################
# Audio-driven upper-body motion synthesis on a humanoid robot
# Computer Science Tripos Part III Project
# Jan Ondras (jo356@cam.ac.uk), Trinity College, University of Cambridge
# 2017/18
#############################################################################################################
# MaryTTS client module, 
# based on https://github.com/marytts/marytts-txt2wav/blob/python/txt2wav.py
#############################################################################################################

import httplib2
from urllib import urlencode # For URL creation

def txt2wav(input_text, audio_path, voice):

    # Mary server informations
    # Remote web client at
    # http://mary.dfki.de:59125/
    # mary_host = "mary.dfki.de"
    # mary_port = "59125"
    # If local server, run by ./marytts-server
    mary_host = "localhost"
    mary_port = "59125"
    
    if voice == 'OB':
        voice_type = "dfki-obadiah-hsmm" # HMM-based model
    elif voice == 'SP':
        voice_type = "dfki-spike-hsmm"
    elif voice == 'PR':
        voice_type = "dfki-prudence-hsmm"
    elif voice == 'PO':
        voice_type = "dfki-poppy-hsmm"
    else:
        raise ValueError("Unknown voice type requested!")

    # Build the query
    query_hash = {"INPUT_TEXT":input_text,
                  "INPUT_TYPE":"TEXT",   # Input text
                  "LOCALE":"en_GB",      # or en_US, ...
                  "VOICE":voice_type,
                  "OUTPUT_TYPE":"AUDIO",
                  "AUDIO":"WAVE", # Audio informations (need both)
                  }
    print "VOICE: ", query_hash["VOICE"], "\n"
    query = urlencode(query_hash)
    print "QUERY = \"http://%s:%s/process?%s\"" % (mary_host, mary_port, query)

    # Run the query to mary http server
    h_mary = httplib2.Http()
    resp, content = h_mary.request("http://%s:%s/process?" % (mary_host, mary_port), "POST", query)

    #  Decode the wav file or raise an exception if no wav files
    if (resp["content-type"] == "audio/x-wav"):

        # Write the wav file
        f = open(audio_path, "wb")
        f.write(content)
        f.close()

    else:
        raise Exception(content)
    print "Wav file saved."
    