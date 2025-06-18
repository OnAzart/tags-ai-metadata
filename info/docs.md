  🔧 How JSON Input Mode Works

  The JSON input mode allows you to process multiple columns efficiently by providing structured data instead of using the
  form interface.

  Supported JSON Formats:

  Format 1: Array of Objects (Recommended)

  [
    {
      "name": "user_email",
      "description": "Email address of the registered user"
    },
    {
      "name": "purchase_amount",
      "description": "Total amount spent on the purchase in USD"
    },
    {
      "name": "birth_date",
      "description": "Date of birth of the customer"
    }
  ]

  Format 2: Nested Object with Columns Array

  {
    "columns": [
      {
        "name": "user_email",
        "description": "Email address of the registered user"
      },
      {
        "name": "purchase_amount",
        "description": "Total amount spent on the purchase in USD"
      }
    ]
  }

  Format 3: Simple Key-Value Pairs

  {
    "user_email": "Email address of the registered user",
    "purchase_amount": "Total amount spent on the purchase in USD",
    "birth_date": "Date of birth of the customer"
  }

  Flexible Field Names:

  The parser supports multiple field name variations:
  - Column Name: name, column_name
  - Description: description, desc, column_description

  Processing Workflow:

  1. Input Validation: JSON is parsed and validated for structure
  2. Data Extraction: Column names and descriptions are extracted regardless of format
  3. Mode Selection: Choose between individual or batch processing
  4. Analysis: Columns are processed using the selected method
  5. Results Display: Same UI as form mode with timing information

  Advantages of JSON Mode:

  Bulk Processing 📈

  - Process dozens of columns at once
  - Copy-paste from spreadsheets or databases
  - Programmatic integration capability

  Data Consistency ✅

  - Structured format reduces input errors
  - Consistent field naming
  - Easy validation and preprocessing

  Integration Ready 🔌

  - API-friendly input format
  - Database export compatible
  - Scriptable and automatable

  Performance Benefits ⚡

  - No form field limitations
  - Faster input for many columns
  - Combined with batch processing = maximum efficiency

  Best Practices:

  1. Use Format 1 for simplicity and readability
  2. Enable Batch Processing for multiple columns
  3. Validate JSON before submission (syntax highlighting helps)
  4. Keep descriptions descriptive for better AI analysis
  5. Use consistent naming across your datasets