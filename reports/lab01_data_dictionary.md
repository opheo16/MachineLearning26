# Lab 01: Data Dictionary

### 1. `customers` Table
| Column | Data Type | Description | Example Value |
| :--- | :--- | :--- | :--- |
| `customer_id` | String | Unique ID for each customer order | `06b8999e2fba1a1fbc88172c00ba8bc7` |
| `customer_unique_id` | String | Unique ID identifying the actual person | `861eff4711a542e4b93843c6dd7febb0` |
| `customer_zip_code_prefix` | Integer | First 5 digits of customer zip code | `14409` |
| `customer_city` | String | City name where customer lives | `franca` |
| `customer_state` | String | Two-letter state code | `SP` |

### 2. `orders` Table
| Column | Data Type | Description | Example Value |
| :--- | :--- | :--- | :--- |
| `order_id` | String | Unique identifier of each order | `e481f51cbdc536705058c14d28b06169` |
| `customer_id` | String | Key linking to the customer table | `9ef443d4ab7c0474276f1d73b0857551` |
| `order_status` | String | Status (`delivered`, `shipped`, `canceled`) | `delivered` |
| `order_purchase_timestamp` | String/Date | Timestamp when purchase was made | `2017-10-02 10:56:33` |
| `order_delivered_customer_date` | String/Date | Date order reached customer | `2017-10-10 21:25:13` |

### 3. `order_items` Table
| Column | Data Type | Description | Example Value |
| :--- | :--- | :--- | :--- |
| `order_id` | String | Order identifier | `00010242fe8c5a6d1ba2dd792cb16214` |
| `order_item_id` | Integer | Sequential number identifying items in same order | `1` |
| `product_id` | String | Unique product identifier | `424473e6550844792f11a49725998b5b` |
| `seller_id` | String | Unique seller identifier | `48436984e03b75fa490e663a94a20b08` |
| `price` | Float | Item selling price | `58.90` |
| `freight_value` | Float | Shipping cost for the item | `13.29` |