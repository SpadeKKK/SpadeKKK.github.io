#!/bin/bash

# A/B Test Designer Robot - Complete Deployment Script
# Supports multiple deployment platforms with comprehensive error handling

set -e  # Exit on any error
set -u  # Exit on undefined variable

# Script metadata
SCRIPT_VERSION="1.0.0"
SCRIPT_NAME="A/B Test Designer Deployment"

# Colors for enhanced output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly PURPLE='\033[0;35m'
readonly CYAN='\033[0;36m'
readonly NC='\033[0m' # No Color

# Configuration - MODIFY THESE FOR YOUR SETUP
DOMAIN_NAME="${DOMAIN_NAME:-AB_demo.io}"
APP_NAME="${APP_NAME:-ab-test-designer}"
HEROKU_APP_NAME="${HEROKU_APP_NAME:-ab-demo-io}"
DOCKER_IMAGE_NAME="${DOCKER_IMAGE_NAME:-ab-test-designer}"
GCP_REGION="${GCP_REGION:-us-central1}"

# Runtime configuration
VERBOSE="${VERBOSE:-false}"
DRY_RUN="${DRY_RUN:-false}"
SKIP_TESTS="${SKIP_TESTS:-false}"
DEPLOYMENT_ENV="${DEPLOYMENT_ENV:-production}"

# Logging function
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    case "$level" in
        "INFO")  echo -e "${GREEN}[INFO]${NC} [$timestamp] $message" ;;
        "WARN")  echo -e "${YELLOW}[WARN]${NC} [$timestamp] $message" ;;
        "ERROR") echo -e "${RED}[ERROR]${NC} [$timestamp] $message" >&2 ;;
        "DEBUG") [[ "$VERBOSE" == "true" ]] && echo -e "${CYAN}[DEBUG]${NC} [$timestamp] $message" ;;
        "SUCCESS") echo -e "${GREEN}[SUCCESS]${NC} [$timestamp] $message" ;;
        *)       echo "[$timestamp] $message" ;;
    esac
}

# Function to print the header
print_header() {
    echo -e "${BLUE}"
    cat << 'EOF'
╔══════════════════════════════════════════════════════════════╗
║                A/B Test Designer Robot                      ║
║            Intelligent Statistical Analysis Platform        ║
║                                                             ║
║                    🚀 DEPLOYMENT SCRIPT 🚀                  ║
╚══════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
    echo -e "${PURPLE}Version: ${SCRIPT_VERSION}${NC}"
    echo -e "${PURPLE}Target Domain: ${DOMAIN_NAME}${NC}"
    echo -e "${PURPLE}Environment: ${DEPLOYMENT_ENV}${NC}"
    echo
}

# Function to check command availability
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check system requirements
check_requirements() {
    log "INFO" "Checking system requirements..."
    
    local required_commands=("git" "python3" "pip")
    local optional_commands=("docker" "npm" "node" "curl")
    local missing_required=()
    local missing_optional=()
    
    # Check required commands
    for cmd in "${required_commands[@]}"; do
        if ! command_exists "$cmd"; then
            missing_required+=("$cmd")
        fi
    done
    
    # Check optional commands
    for cmd in "${optional_commands[@]}"; do
        if ! command_exists "$cmd"; then
            missing_optional+=("$cmd")
        fi
    done
    
    # Report missing required commands
    if [ ${#missing_required[@]} -ne 0 ]; then
        log "ERROR" "Missing required commands: ${missing_required[*]}"
        log "ERROR" "Please install the missing tools and try again."
        echo
        echo "Installation guides:"
        echo "  Git: https://git-scm.com/downloads"
        echo "  Python: https://python.org/downloads"
        exit 1
    fi
    
    # Report missing optional commands
    if [ ${#missing_optional[@]} -ne 0 ]; then
        log "WARN" "Missing optional commands: ${missing_optional[*]}"
        log "WARN" "Some deployment options may not be available."
    fi
    
    # Check Python version
    local python_version
    python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
    log "INFO" "Python version: $python_version"
    
    # Check Git repository
    if [ -d ".git" ]; then
        local git_branch
        git_branch=$(git branch --show-current 2>/dev/null || echo "unknown")
        log "INFO" "Git repository detected (branch: $git_branch)"
    else
        log "WARN" "Not in a Git repository. Some features may not work."
    fi
    
    log "SUCCESS" "System requirements check completed"
}

# Function to validate project structure
validate_project_structure() {
    log "INFO" "Validating project structure..."
    
    local required_files=("app.py" "requirements.txt" "index.html")
    local recommended_files=("statistical_engine.py" "data_generator.py" "README.md")
    local missing_required=()
    local missing_recommended=()
    
    # Check required files
    for file in "${required_files[@]}"; do
        if [ ! -f "$file" ]; then
            missing_required+=("$file")
        fi
    done
    
    # Check recommended files
    for file in "${recommended_files[@]}"; do
        if [ ! -f "$file" ]; then
            missing_recommended+=("$file")
        fi
    done
    
    if [ ${#missing_required[@]} -ne 0 ]; then
        log "ERROR" "Missing required files: ${missing_required[*]}"
        log "ERROR" "Please ensure you're running this script from the project root directory."
        exit 1
    fi
    
    if [ ${#missing_recommended[@]} -ne 0 ]; then
        log "WARN" "Missing recommended files: ${missing_recommended[*]}"
        log "WARN" "Some features may not work optimally."
    fi
    
    log "SUCCESS" "Project structure validation completed"
}

# Function to setup local development environment
setup_local_environment() {
    log "INFO" "Setting up local development environment..."
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        log "INFO" "Creating Python virtual environment..."
        if [[ "$DRY_RUN" == "false" ]]; then
            python3 -m venv venv
        fi
    else
        log "INFO" "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    if [[ "$DRY_RUN" == "false" ]]; then
        log "INFO" "Activating virtual environment..."
        # shellcheck source=/dev/null
        source venv/bin/activate
        
        # Upgrade pip
        log "INFO" "Upgrading pip..."
        python -m pip install --upgrade pip
        
        # Install dependencies
        log "INFO" "Installing Python dependencies..."
        pip install -r requirements.txt
        
        # Install development dependencies
        log "INFO" "Installing development dependencies..."
        pip install pytest pytest-cov flake8 black bandit safety
    fi
    
    log "SUCCESS" "Local environment setup completed"
}

# Function to run code quality checks
run_quality_checks() {
    log "INFO" "Running code quality checks..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "DRY RUN: Would run quality checks"
        return 0
    fi
    
    # Activate virtual environment
    # shellcheck source=/dev/null
    source venv/bin/activate
    
    # Run linting
    log "INFO" "Running flake8 linter..."
    if flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics; then
        log "SUCCESS" "Linting passed"
    else
        log "WARN" "Linting found issues (continuing anyway)"
    fi
    
    # Run security checks
    log "INFO" "Running security analysis..."
    if bandit -r . -x tests/ -f json -o bandit-report.json >/dev/null 2>&1; then
        log "SUCCESS" "Security analysis passed"
    else
        log "WARN" "Security analysis found potential issues"
    fi
    
    # Check for known vulnerabilities
    log "INFO" "Checking for known vulnerabilities..."
    if safety check --json >/dev/null 2>&1; then
        log "SUCCESS" "Vulnerability check passed"
    else
        log "WARN" "Potential vulnerabilities found in dependencies"
    fi
}

# Function to run tests
run_tests() {
    if [[ "$SKIP_TESTS" == "true" ]]; then
        log "INFO" "Skipping tests (SKIP_TESTS=true)"
        return 0
    fi
    
    log "INFO" "Running test suite..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "DRY RUN: Would run tests"
        return 0
    fi
    
    # Check if virtual environment exists
    if [ ! -d "venv" ]; then
        setup_local_environment
    fi
    
    # Activate virtual environment
    # shellcheck source=/dev/null
    source venv/bin/activate
    
    # Create tests directory if it doesn't exist
    if [ ! -d "tests" ]; then
        log "INFO" "Creating tests directory..."
        mkdir -p tests
        touch tests/__init__.py
    fi
    
    # Run tests if test files exist
    if find tests/ -name "test_*.py" -type f | grep -q .; then
        log "INFO" "Running pytest..."
        if pytest tests/ -v --cov=. --cov-report=term-missing --cov-report=html; then
            log "SUCCESS" "All tests passed"
        else
            log "ERROR" "Tests failed"
            if [[ "$DEPLOYMENT_ENV" == "production" ]]; then
                log "ERROR" "Cannot deploy to production with failing tests"
                exit 1
            else
                log "WARN" "Continuing despite test failures (non-production environment)"
            fi
        fi
    else
        log "WARN" "No test files found, skipping test execution"
    fi
}

# Function to build Docker image
build_docker_image() {
    log "INFO" "Building Docker image..."
    
    if ! command_exists docker; then
        log "WARN" "Docker not found. Skipping Docker image build."
        return 0
    fi
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "DRY RUN: Would build Docker image: ${DOCKER_IMAGE_NAME}:latest"
        return 0
    fi
    
    # Build Docker image with BuildKit for better performance
    log "INFO" "Building image: ${DOCKER_IMAGE_NAME}:latest"
    if DOCKER_BUILDKIT=1 docker build \
        --tag "${DOCKER_IMAGE_NAME}:latest" \
        --tag "${DOCKER_IMAGE_NAME}:$(date +%Y%m%d-%H%M%S)" \
        --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
        --build-arg VERSION="$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')" \
        .; then
        log "SUCCESS" "Docker image built successfully"
        
        # Show image size
        local image_size
        image_size=$(docker images "${DOCKER_IMAGE_NAME}:latest" --format "table {{.Size}}" | tail -n 1)
        log "INFO" "Image size: $image_size"
    else
        log "ERROR" "Docker build failed"
        return 1
    fi
}

# Function to run local development server
run_local_server() {
    log "INFO" "Starting local development server..."
    
    # Setup environment if needed
    if [ ! -d "venv" ]; then
        setup_local_environment
    fi
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "DRY RUN: Would start local server at http://localhost:5000"
        return 0
    fi
    
    # Activate virtual environment
    # shellcheck source=/dev/null
    source venv/bin/activate
    
    log "INFO" "Server will be available at: http://localhost:5000"
    log "INFO" "Press Ctrl+C to stop the server"
    
    # Set environment variables
    export FLASK_ENV=development
    export FLASK_DEBUG=true
    
    # Start the server
    python app.py
}

# Function to deploy to Vercel
deploy_to_vercel() {
    log "INFO" "Deploying to Vercel..."
    
    if ! command_exists npm; then
        log "ERROR" "Node.js/npm not found. Please install Node.js to use Vercel deployment."
        return 1
    fi
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "DRY RUN: Would deploy to Vercel"
        return 0
    fi
    
    # Install Vercel CLI if not present
    if ! command_exists vercel; then
        log "INFO" "Installing Vercel CLI..."
        npm install -g vercel@latest
    fi
    
    # Deploy to Vercel
    log "INFO" "Starting Vercel deployment..."
    if vercel --prod --confirm; then
        log "SUCCESS" "Vercel deployment completed"
        
        # Setup custom domain if specified
        if [[ "$DOMAIN_NAME" != "AB_demo.io" ]] && [[ "$DOMAIN_NAME" != "localhost" ]]; then
            log "INFO" "Setting up custom domain: $DOMAIN_NAME"
            if vercel domains add "$DOMAIN_NAME" 2>/dev/null; then
                log "SUCCESS" "Custom domain added: $DOMAIN_NAME"
            else
                log "WARN" "Custom domain may already exist or require DNS configuration"
            fi
        fi
        
        log "INFO" "🌐 Your application should be available at: https://$DOMAIN_NAME"
    else
        log "ERROR" "Vercel deployment failed"
        return 1
    fi
}

# Function to deploy to Heroku
deploy_to_heroku() {
    log "INFO" "Deploying to Heroku..."
    
    if ! command_exists heroku; then
        log "ERROR" "Heroku CLI not found. Please install it from https://devcenter.heroku.com/articles/heroku-cli"
        return 1
    fi
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "DRY RUN: Would deploy to Heroku app: $HEROKU_APP_NAME"
        return 0
    fi
    
    # Login check
    if ! heroku auth:whoami >/dev/null 2>&1; then
        log "INFO" "Please log in to Heroku..."
        heroku login
    fi
    
    # Create app if it doesn't exist
    if ! heroku apps:info "$HEROKU_APP_NAME" >/dev/null 2>&1; then
        log "INFO" "Creating Heroku app: $HEROKU_APP_NAME"
        heroku create "$HEROKU_APP_NAME"
    fi
    
    # Set buildpack
    log "INFO" "Setting Python buildpack..."
    heroku buildpacks:set heroku/python --app "$HEROKU_APP_NAME"
    
    # Set environment variables
    log "INFO" "Setting environment variables..."
    heroku config:set FLASK_ENV=production --app "$HEROKU_APP_NAME"
    heroku config:set PYTHONPATH=/app --app "$HEROKU_APP_NAME"
    
    # Deploy using Git
    log "INFO" "Deploying to Heroku..."
    
    # Add Heroku remote if it doesn't exist
    if ! git remote | grep -q heroku; then
        heroku git:remote -a "$HEROKU_APP_NAME"
    fi
    
    # Push to Heroku
    if git push heroku main; then
        log "SUCCESS" "Heroku deployment completed"
        
        # Add custom domain
        if [[ "$DOMAIN_NAME" != "AB_demo.io" ]] && [[ "$DOMAIN_NAME" != "localhost" ]]; then
            log "INFO" "Adding custom domain: $DOMAIN_NAME"
            if heroku domains:add "$DOMAIN_NAME" --app "$HEROKU_APP_NAME" 2>/dev/null; then
                log "SUCCESS" "Custom domain added"
            else
                log "WARN" "Custom domain may already exist"
            fi
        fi
        
        log "INFO" "🌐 App URL: https://$HEROKU_APP_NAME.herokuapp.com"
    else
        log "ERROR" "Heroku deployment failed"
        return 1
    fi
}

# Function to deploy to Google Cloud Run
deploy_to_gcp() {
    log "INFO" "Deploying to Google Cloud Run..."
    
    if ! command_exists gcloud; then
        log "ERROR" "Google Cloud CLI not found. Please install it from https://cloud.google.com/sdk"
        return 1
    fi
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "DRY RUN: Would deploy to Google Cloud Run"
        return 0
    fi
    
    # Check authentication
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -n 1 >/dev/null 2>&1; then
        log "INFO" "Please authenticate with Google Cloud..."
        gcloud auth login
    fi
    
    # Get project ID
    local project_id
    project_id=$(gcloud config get-value project 2>/dev/null)
    if [ -z "$project_id" ]; then
        log "ERROR" "No GCP project set. Please run 'gcloud config set project YOUR_PROJECT_ID'"
        return 1
    fi
    
    log "INFO" "Using GCP project: $project_id"
    
    # Enable required APIs
    log "INFO" "Enabling required APIs..."
    gcloud services enable cloudbuild.googleapis.com run.googleapis.com
    
    # Build and deploy using Cloud Build
    log "INFO" "Building and deploying with Cloud Build..."
    if gcloud run deploy "$APP_NAME" \
        --source . \
        --platform managed \
        --region "$GCP_REGION" \
        --allow-unauthenticated \
        --port 5000 \
        --memory 2Gi \
        --cpu 2 \
        --max-instances 10 \
        --set-env-vars "FLASK_ENV=production"; then
        
        log "SUCCESS" "Google Cloud Run deployment completed"
        
        # Get service URL
        local service_url
        service_url=$(gcloud run services describe "$APP_NAME" \
            --platform managed \
            --region "$GCP_REGION" \
            --format 'value(status.url)')
        
        log "INFO" "🌐 Service URL: $service_url"
    else
        log "ERROR" "Google Cloud Run deployment failed"
        return 1
    fi
}

# Function to setup GitHub repository
setup_github_repository() {
    log "INFO" "Setting up GitHub repository..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "DRY RUN: Would setup GitHub repository"
        return 0
    fi
    
    # Check if this is already a git repository
    if [ ! -d ".git" ]; then
        log "INFO" "Initializing Git repository..."
        git init
        git add .
        git commit -m "Initial commit: A/B Test Designer Robot"
    fi
    
    # Show GitHub setup instructions
    log "INFO" "GitHub repository setup instructions:"
    echo
    echo "1. Create a new repository on GitHub"
    echo "2. Run the following commands:"
    echo "   git remote add origin https://github.com/YOUR_USERNAME/ab-test-designer.git"
    echo "   git branch -M main"
    echo "   git push -u origin main"
    echo
    echo "3. Set up GitHub Secrets for CI/CD:"
    echo "   - VERCEL_TOKEN"
    echo "   - DOCKERHUB_USERNAME"
    echo "   - DOCKERHUB_TOKEN"
    echo "   - HEROKU_API_KEY (if using Heroku)"
    echo "   - GCP_SA_KEY (if using Google Cloud)"
    echo
    
    log "SUCCESS" "GitHub setup instructions provided"
}

# Function to show deployment menu
show_deployment_menu() {
    echo
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                    Deployment Options                       ║${NC}"
    echo -e "${CYAN}╠══════════════════════════════════════════════════════════════╣${NC}"
    echo -e "${CYAN}║  ${YELLOW}1)${NC} Setup local development environment            ${CYAN}║${NC}"
    echo -e "${CYAN}║  ${YELLOW}2)${NC} Run local development server                  ${CYAN}║${NC}"
    echo -e "${CYAN}║  ${YELLOW}3)${NC} Run tests and quality checks                  ${CYAN}║${NC}"
    echo -e "${CYAN}║  ${YELLOW}4)${NC} Build Docker image                            ${CYAN}║${NC}"
    echo -e "${CYAN}║  ${YELLOW}5)${NC} Setup GitHub repository                       ${CYAN}║${NC}"
    echo -e "${CYAN}║                                                      ║${NC}"
    echo -e "${CYAN}║        ${GREEN}--- Cloud Deployment Platforms ---${NC}              ${CYAN}║${NC}"
    echo -e "${CYAN}║  ${YELLOW}6)${NC} Deploy to Vercel (Recommended for ${DOMAIN_NAME})    ${CYAN}║${NC}"
    echo -e "${CYAN}║  ${YELLOW}7)${NC} Deploy to Heroku                              ${CYAN}║${NC}"
    echo -e "${CYAN}║  ${YELLOW}8)${NC} Deploy to Google Cloud Run                   ${CYAN}║${NC}"
    echo -e "${CYAN}║                                                      ║${NC}"
    echo -e "${CYAN}║  ${YELLOW}9)${NC} Full deployment pipeline (recommended)       ${CYAN}║${NC}"
    echo -e "${CYAN}║  ${YELLOW}0)${NC} Exit                                          ${CYAN}║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo
}

# Function to run full deployment pipeline
run_full_deployment_pipeline() {
    log "INFO" "Starting full deployment pipeline..."
    
    # Pre-deployment checks
    check_requirements
    validate_project_structure
    
    # Setup and testing
    setup_local_environment
    run_quality_checks
    run_tests
    
    # Build
    build_docker_image
    
    # Choose deployment platform
    echo
    log "INFO" "Choose deployment platform:"
    echo "  1) Vercel (Recommended for ${DOMAIN_NAME})"
    echo "  2) Heroku"
    echo "  3) Google Cloud Run"
    echo
    
    local platform_choice
    read -p "Enter your choice (1-3): " platform_choice
    
    case $platform_choice in
        1)
            deploy_to_vercel
            ;;
        2)
            deploy_to_heroku
            ;;
        3)
            deploy_to_gcp
            ;;
        *)
            log "ERROR" "Invalid choice"
            return 1
            ;;
    esac
    
    # Post-deployment
    log "SUCCESS" "Full deployment pipeline completed! 🎉"
    echo
    log "INFO" "🌐 Your A/B Test Designer should be available at: https://${DOMAIN_NAME}"
    echo
    log "INFO" "Next steps:"
    echo "  1. Test your deployed application"
    echo "  2. Configure DNS settings if using a custom domain"
    echo "  3. Set up monitoring and analytics"
    echo "  4. Review security settings"
    echo
    log "WARN" "Don't forget to configure DNS settings for custom domains!"
}

# Function to handle script arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -v|--verbose)
                VERBOSE="true"
                shift
                ;;
            -d|--dry-run)
                DRY_RUN="true"
                shift
                ;;
            --skip-tests)
                SKIP_TESTS="true"
                shift
                ;;
            --domain)
                DOMAIN_NAME="$2"
                shift 2
                ;;
            --env)
                DEPLOYMENT_ENV="$2"
                shift 2
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                log "ERROR" "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# Function to show help
show_help() {
    cat << EOF
A/B Test Designer Robot - Deployment Script

Usage: $0 [OPTIONS]

Options:
    -v, --verbose       Enable verbose output
    -d, --dry-run       Perform a dry run (no actual changes)
    --skip-tests        Skip running tests
    --domain DOMAIN     Set custom domain name (default: $DOMAIN_NAME)
    --env ENV           Set deployment environment (default: $DEPLOYMENT_ENV)
    -h, --help          Show this help message

Environment Variables:
    DOMAIN_NAME         Custom domain name
    VERBOSE             Enable verbose logging
    DRY_RUN             Enable dry run mode
    SKIP_TESTS          Skip test execution

Examples:
    $0                          # Interactive mode
    $0 --verbose --domain myapp.com
    $0 --dry-run --skip-tests
    
For more information, visit: https://github.com/spadekkk/ab-test-designer
EOF
}

# Function to cleanup on exit
cleanup() {
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        log "ERROR" "Script failed with exit code $exit_code"
    fi
    
    # Remove temporary files
    rm -f bandit-report.json 2>/dev/null || true
    
    exit $exit_code
}

# Set up signal handling
trap cleanup EXIT
trap 'log "ERROR" "Script interrupted"; exit 130' INT TERM

# Main function
main() {
    # Parse command line arguments
    parse_arguments "$@"
    
    # Print header
    print_header
    
    # Make script executable if needed
    if [ ! -x "$0" ]; then
        log "WARN" "Making script executable..."
        chmod +x "$0"
    fi
    
    # If arguments provided, run non-interactively
    if [[ "$DRY_RUN" == "true" ]] || [[ "$VERBOSE" == "true" ]]; then
        run_full_deployment_pipeline
        return $?
    fi
    
    # Interactive mode
    while true; do
        show_deployment_menu
        local choice
        read -p "Enter your choice (0-9): " choice
        echo
        
        case $choice in
            1)
                check_requirements && validate_project_structure && setup_local_environment
                ;;
            2)
                run_local_server
                ;;
            3)
                run_quality_checks && run_tests
                ;;
            4)
                build_docker_image
                ;;
            5)
                setup_github_repository
                ;;
            6)
                deploy_to_vercel
                ;;
            7)
                deploy_to_heroku
                ;;
            8)
                deploy_to_gcp
                ;;
            9)
                run_full_deployment_pipeline
                ;;
            0)
                log "SUCCESS" "Goodbye! 👋"
                exit 0
                ;;
            *)
                log "ERROR" "Invalid choice. Please enter 0-9."
                ;;
        esac
        
        echo
        read -p "Press Enter to continue..."
        echo
    done
}

# Run main function with all arguments
main "$@"