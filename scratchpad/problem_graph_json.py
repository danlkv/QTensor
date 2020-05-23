from PyInquirer import style_from_dict, Token, prompt
from PyInquirer import Validator, ValidationError
import fire


style = style_from_dict({
    Token.QuestionMark: '#E91E63 bold',
    Token.Selected: '#673AB7 bold',
    Token.Instruction: '',  # default
    Token.Answer: '#2196f3 bold',
    Token.Question: '',
})



class NumberValidator(Validator):
    def validate(self, document):
        try:
            int(document.text)
        except ValueError:
            raise ValidationError(
                message='Please enter a number',
                cursor_position=len(document.text))  # Move cursor to end


print('Hi, welcome to Python Pizza')

questions = [
    dict(
        type='confirm'
        ,name='delivery'
        ,message='Is this delivery?'
        ,default=False
    )
    ,dict(type='input'
          ,name='age'
          ,message='What is your age?'
          ,validate=NumberValidator
         )
    ,dict(type='list'
          ,name='graph_type'
          ,message='Enter type of graph'
          ,choices=['random', 'randomreg']
         )
    ,dict(type='input'
          ,name='size'
          ,message='Size of the problem'
          ,validate=NumberValidator
         ),

    {
        'type': 'list',
        'name': 'prize',
        'message': 'For leaving a comment, you get a freebie',
        'choices': ['cake', 'fries'],
        'when': lambda answers: answers['comments'] != 'Nope, all good!'
    }
]

class Graph:

    @classmethod
    def to_json(cls, size, type, degree=3):
        types_allowed = ['randomreg', 'grid', 'rectgrid', 'randomgnp']
        if type in types_allowed:
            args = dict(size=size, degree=degree)
        else:
            raise Exception(f"Invalid graph type {type}, should be one of {types_allowed}")
fire.Fire(Graph)


"""
answers = prompt(questions, style=style)
print('Order receipt:')
print(answers)
"""
