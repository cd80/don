# Don Trading Framework Configuration

## Environment Variables
The framework requires the following environment variables to be set:

### Required Variables
```bash
# Binance API Configuration
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret

# Database Configuration
DATABASE_URL=postgresql://user:password@host:port/dbname
```

### Optional Variables
```bash
# Logging Configuration
LOG_LEVEL=INFO  # or DEBUG for verbose output

# Feature Calculation Settings
FEATURE_WINDOW=14  # Technical indicator window size
```

## Database Setup
The framework requires a PostgreSQL database:

1. Create a new database:
```sql
CREATE DATABASE don;
```

2. Set up the database URL:
```bash
export DATABASE_URL="postgresql://user:password@localhost:5432/don"
```

3. Tables will be automatically created during `don setup --all`

## Binance API Configuration
1. Create API keys from Binance Futures:
   - Visit Binance Futures dashboard
   - Generate API key and secret
   - Enable futures trading permissions

2. Set environment variables:
```bash
export BINANCE_API_KEY="your_api_key"
export BINANCE_API_SECRET="your_api_secret"
```

## Configuration Validation
Run setup to validate configuration:
```bash
don setup --all
```

This will:
- Check environment variables
- Validate API keys
- Test database connection
- Initialize required tables

## Security Notes
- Never commit API keys to version control
- Use `.env` files for local development
- Ensure proper database user permissions
- Regularly rotate API keys

## Troubleshooting
Common configuration issues:

1. Database Connection:
   - Check PostgreSQL service is running
   - Verify database exists
   - Confirm user permissions

2. API Keys:
   - Ensure keys have correct permissions
   - Verify keys are active
   - Check for proper encoding

3. Environment Variables:
   - Verify all required variables are set
   - Check for proper formatting
   - Validate file permissions for .env
