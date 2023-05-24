from django import forms

class PredictForm(forms.Form):
    to_predict_field = forms.CharField(label="Prediction Sentence:", 
        widget=forms.Textarea(attrs={'name':'body', 'rows':3, 'cols':66}), required=False)

    def get_user_input(self):
        return self.data["to_predict_field"]

    def get_sanitized_input(self):
        user_input = self.get_user_input()
        all_lowers = str.lower(user_input)
        return all_lowers

    def is_valid(self):
        text = self.data["to_predict_field"]
        # Check if field empty
        if text == "":
            return False
        # Text processing is now handled by the neural net. No need to pre-filter.
        return True

class LoginForm(forms.Form):
    user_field = forms.CharField(label='Username\n')
    pass_field = forms.CharField(label='Password\n', widget=forms.PasswordInput)

    def grab_credentials(self):
        username = self.data["user_field"]
        password = self.data["pass_field"]
        # sanitize
        return username, password