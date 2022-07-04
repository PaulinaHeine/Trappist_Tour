""" Submission helper to generate a .json-file for submission to optimize.esa.int. """
import json
import numpy

def create_submission(challenge_id, problem_id, x, fn_out = './submission.json', name = '', description= ''):
    """ The following parameters are mandatory to create a submission file:

        challenge_id: a string of the challenge identifier (found on the corresponding problem page)
        problem_id: a string of the problem identifier (found on the corresponding problem page)
        x: for single-objective problems: a list of numbers determining the decision vector
           for multi-objective problems: a list of list of numbers determining a population of decision vectors

        Optionally provide:
        fn_out: a string indicating the output path and filename
        name: a string that can be used to give your submission a title
        description: a string that can contain meta-information about your submission
    """
    assert type(challenge_id) == str
    assert type(problem_id) == str
    assert type(x) in [list, numpy.ndarray]
    assert type(fn_out) == str
    assert type(name) == str
    assert type(description) == str

    # converting numpy datatypes to python datatypes
    x = numpy.array(x).tolist()

    d = {'decisionVector':x,
         'problem':problem_id,
         'challenge':challenge_id,
         'name':name,
         'description':description }

    with open(fn_out, 'wt') as json_file:
        json.dump([d], json_file, indent = 6)