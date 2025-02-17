# Installation

## Prerequisites

- **Python 3.10+**
- **PostgreSQL** (for Prefect Orion database)

### Setting up PostgreSQL

1. **Install PostgreSQL**:

   - **macOS (Homebrew)**:

     ```bash
     brew install postgresql
     brew services start postgresql
     ```

   - **Linux (Debian/Ubuntu)**:

     ```bash
     sudo apt-get update && sudo apt-get install -y postgresql postgresql-contrib
     sudo service postgresql start
     ```

   - **Windows**:
     Download and run the [PostgreSQL installer](https://www.postgresql.org/download/windows/) and follow the prompts.

2. **Create a PostgreSQL User and Database**:

   ```bash
   psql postgres
   ```

   - **Windows**:
   Download and run the [PostgreSQL installer](https://www.postgresql.org/download/windows/) and follow the prompts.

2. **Create a PostgreSQL User and Database**:

   ```bash
   psql postgres
   ```

   Inside the `psql` prompt:

   ```sql
   CREATE USER prefect_user WITH PASSWORD 'pass';
   CREATE DATABASE prefect_db OWNER prefect_user;
   GRANT ALL PRIVILEGES ON DATABASE prefect_db TO prefect_user;
   ```

   Exit with `\q`.

### Installing nxbench

From PyPi:

```bash
pip install nxbench
```

From a local clone:

```bash
git clone https://github.com/dpys/nxbench.git
cd nxbench
make install
```

### Setting up Prefect Orion

In a terminal window:

1. **Export environment variables pointing to your PostgreSQL database**:

   ```bash
   export PREFECT_API_DATABASE_CONNECTION_URL="postgresql+asyncpg://prefect_user:pass@localhost:5432/prefect_db"
   export PREFECT_ORION_DATABASE_CONNECTION_POOL_SIZE="5"
   export PREFECT_ORION_DATABASE_CONNECTION_MAX_OVERFLOW="10"
   export PREFECT_ORION_API_ENABLE_TASK_RUN_DATA_PERSISTENCE="false"
   export PREFECT_API_URL="<http://127.0.0.1:4200/api>"
   ```

2. **Start the Orion server**:

   ```bash
   prefect orion start
   ```

   By default it will run on `http://127.0.0.1:4200`.

## Installation (Docker Setup)

```bash
# CPU-only
docker-compose -f docker/docker-compose.cpu.yaml build

# With GPU
docker-compose -f docker/docker-compose.gpu.yaml build
```

## Development Setup

Install development dependencies (testing and documentation):

```bash
pip install -e .[test,doc]
```

Run tests:

```bash
make test
```
