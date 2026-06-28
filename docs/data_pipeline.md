I - Scope and Purpose

- Build a reliable data ingestion and storage pipeline for market-related data
- Focus on correctes, debuggability and robustness rather than performance
- This pipeline is a foundation for future pricing models

II - Data Sources :

- External data source accessed via API 
- Data pulled at regular intervals (bacth-style, not streaming)
- Data assumed to be mostly correct but subject to :
  - network failures
  - incomplete responses
  - occasional malformed entries

III - High level architecture

- Data fetched from API
- Raw data validated and cleaned
- Data stored locally in structured format
- Stored data later consumed by downstream component (future models/UI)

IV - Data flow

- Request data from API
- Check responses status and format
- Parse raw responsed into structured records
- Validate required fields
- Write validated data to storage
- Surface errors clearly when any step fails

V - Storage design

- Data stored in files on disk
- Files organized using Hive-style partitioning
- Partition based on time (e.g. data)
- Goal : make it easy to query subsets of data without scanning
everything

VI - Why Hive-style partitioning

- Suggested as a common industry approach for time-series data
- Helps limit how much data is read for a given query
- Trade-off accepted :
  - More files
  - More complexity in directory structure
- Considered acceptable for learning and experimentatio

VII - Schema Assumptions :

- Structural assumptions :
  - Column use a MultiIndex
  - Level0 = ticker symbol
  - Level1 = field name (e.g. Close, Volume)
  - Row index represents time
- Column-level assumptions :
  - For each ticker, a fixed set of fields exist
  - No missing field level for a given ticker
  - Field names are consistent across tickers
- Type assumptions :
  - Timestamp column is parseable as datatime
  - All dataframe values are numeric
  - No mixed dtypes inside a field
- Consistency assumptions :
  - All columns inside a parquet file correspond to the
  directory's ticker scope (if applicable)
  - No unexpected ticker appears in the MultiIndex
  - No duplicated (ticker, field) column pairs
- Ordering and Completeness :
  - Index may not be sorted when loaded
  - Donwstream code may assumed sorted index

VII - Error handling

- Custom exceptions created for different failure types:
  - API errors
  - Data validation errors
  - Storage write failures
- Errors raised early when assumptions are violated
- Errors designed to be explicit rather than silent