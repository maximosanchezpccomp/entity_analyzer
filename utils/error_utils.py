class SEOAnalyzerError(Exception):
    """Base exception class for SEO Analyzer errors."""
    pass

class URLFetchError(SEOAnalyzerError):
    """Exception raised for errors during URL fetching."""
    def __init__(self, url, message="Error fetching URL content"):
        self.url = url
        self.message = f"{message}: {url}"
        super().__init__(self.message)

class ContentExtractionError(SEOAnalyzerError):
    """Exception raised for errors during content extraction."""
    def __init__(self, url=None, message="Error extracting content"):
        self.url = url
        self.message = message
        if url:
            self.message = f"{message} from {url}"
        super().__init__(self.message)

class NLPProcessingError(SEOAnalyzerError):
    """Exception raised for errors during NLP processing."""
    def __init__(self, operation=None, message="Error during NLP processing"):
        self.operation = operation
        self.message = message
        if operation:
            self.message = f"{message} in {operation}"
        super().__init__(self.message)

class APIKeyError(SEOAnalyzerError):
    """Exception raised for API key related errors."""
    def __init__(self, api_name=None, message="Invalid or missing API key"):
        self.api_name = api_name
        self.message = message
        if api_name:
            self.message = f"{message} for {api_name}"
        super().__init__(self.message)

class InputValidationError(SEOAnalyzerError):
    """Exception raised for invalid user inputs."""
    def __init__(self, input_name=None, message="Invalid input"):
        self.input_name = input_name
        self.message = message
        if input_name:
            self.message = f"{message}: {input_name}"
        super().__init__(self.message)

def format_error_for_user(error):
    """Format exception into user-friendly message."""
    if isinstance(error, URLFetchError):
        return f"Error fetching URL: {error.url}. Please check if the URL is correct and accessible."
    elif isinstance(error, ContentExtractionError):
        return f"Error extracting content: {error.message}. The page might have unusual formatting or restrictions."
    elif isinstance(error, NLPProcessingError):
        return f"Error during analysis: {error.message}. This might be due to content complexity or API limitations."
    elif isinstance(error, APIKeyError):
        return f"API key error: {error.message}. Please check your API key in the sidebar."
    elif isinstance(error, InputValidationError):
        return f"Input error: {error.message}. Please check your input and try again."
    else:
        return f"Unexpected error: {str(error)}. Please try again or contact support if the issue persists."
