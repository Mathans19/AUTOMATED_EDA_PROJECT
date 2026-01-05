# Automated Insights - EDA Application

A powerful Django-based web application for automated Exploratory Data Analysis (EDA) with AI-driven insights, data cleaning, and interactive visualizations.

## 🚀 Features

- **📊 Automated EDA Reports**: Generate comprehensive data profiling reports using ydata_profiling
- **🧹 Data Cleaning**: Interactive data cleaning with multiple operations:
  - Drop columns
  - Handle missing values (drop, fill with mean/median/mode)
  - Remove duplicates
  - Rename columns
  - Type conversion (numeric, datetime, categorical)
  - Outlier removal
- **📈 Interactive Visualizations**: Create beautiful charts using Plotly:
  - Histograms
  - Box plots
  - Scatter plots
  - Correlation heatmaps
- **🤖 AI-Powered Insights**: Ask questions about your data using Groq AI (Llama 3.1)
- **💾 Data Management**: Upload, clean, and download your datasets

## 🛠️ Technology Stack

- **Backend**: Django 4.2.2
- **Database**: MySQL
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, ydata_profiling
- **AI**: Groq API (Llama 3.1)
- **Frontend**: HTML, CSS, JavaScript (with modern glassmorphism UI)

## 📋 Prerequisites

- Python 3.8+
- MySQL Server
- pip (Python package manager)

## 🔧 Installation

1. **Clone the repository** (or navigate to project directory):
   ```bash
   cd AUTOMATED_INSIGHTS
   ```

2. **Create and activate virtual environment**:
   ```bash
   python -m venv myenv
   # On Windows:
   myenv\Scripts\activate
   # On Linux/Mac:
   source myenv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up MySQL database**:
   ```sql
   CREATE DATABASE eda_project;
   ```

5. **Configure environment variables**:
   - Create a `.env` file in the project root (if not already present)
   - Update the values in `.env` with your configuration:
   ```env
   SECRET_KEY=your-secret-key-here
   DEBUG=True
   DB_NAME=eda_project
   DB_USER=root
   DB_PASSWORD=your-password
   DB_HOST=localhost
   DB_PORT=3306
   GROQ_API_KEY=your-groq-api-key-here
   ALLOWED_HOSTS=
   ```

6. **Run migrations**:
   ```bash
   python manage.py makemigrations
   python manage.py migrate
   ```

7. **Create superuser** (optional, for admin access):
   ```bash
   python manage.py createsuperuser
   ```

8. **Run the development server**:
   ```bash
   python manage.py runserver
   ```

9. **Access the application**:
   - Open your browser and navigate to `http://127.0.0.1:8000/`

## 📖 Usage

### Upload Dataset
1. Click on the upload area or drag & drop a CSV file
2. Optionally provide a name for your dataset
3. Click "Start Analysis"

### Generate EDA Report
- After uploading, you'll be redirected to the EDA report page
- The report includes:
  - Dataset overview
  - Variable types and statistics
  - Missing values analysis
  - Correlation analysis
  - Sample data preview

### Clean Data
1. Navigate to the "Clean Data" section
2. View column information (missing values, data types)
3. Select columns and apply cleaning actions:
   - Drop columns
   - Handle missing values
   - Remove duplicates
   - Rename columns
   - Convert data types
   - Remove outliers

### Visualize Data
1. Go to the "Visualize" section
2. Select chart type and columns
3. Generate interactive charts
4. Charts are fully interactive (zoom, pan, hover)

### AI Insights
1. Navigate to "AI Insights"
2. Enter your Groq API key (or set GROQ_API_KEY in environment)
3. Ask questions about your data
4. Get AI-powered answers based on your dataset

### Download Cleaned Data
- After cleaning, download your processed dataset as CSV

## 🔐 Security Notes

- **Never commit `.env` file** to version control
- Use strong `SECRET_KEY` in production
- Set `DEBUG=False` in production
- Configure `ALLOWED_HOSTS` for production deployment
- Use environment variables for all sensitive data

## 📁 Project Structure

```
AUTOMATED_INSIGHTS/
├── eda/                    # Main application
│   ├── models.py          # Database models
│   ├── views.py           # View functions
│   ├── forms.py           # Form definitions
│   ├── urls.py            # URL routing
│   ├── templates/         # HTML templates
│   └── static/            # Static files (CSS, JS)
├── eda_app/               # Django project settings
│   ├── settings.py        # Project settings
│   ├── urls.py            # Root URL configuration
│   └── wsgi.py            # WSGI configuration
├── media/                 # Uploaded files
├── manage.py              # Django management script
├── .env                   # Environment variables (not in git)
└── requirements.txt       # Python dependencies
```

## 🐛 Troubleshooting

### MySQL Connection Issues
- Ensure MySQL server is running
- Verify database credentials in `.env`
- Check if MySQL client library is installed: `pip install mysqlclient`

### File Upload Issues
- Check file size (max 100MB)
- Ensure file is CSV format
- Verify `MEDIA_ROOT` directory permissions

### API Key Issues
- Verify Groq API key is correct
- Check if API key is set in environment or provided in request

## 🚧 Future Enhancements

- [ ] Support for Excel files (.xlsx, .xls)
- [ ] Support for JSON data files
- [ ] Advanced statistical analysis
- [ ] Export reports as PDF
- [ ] User authentication and data isolation
- [ ] Scheduled report generation
- [ ] API endpoints for programmatic access
- [ ] Multi-file comparison analysis

## 📝 License

This project is open source and available for educational purposes.

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Support

For issues and questions, please open an issue on the repository.

