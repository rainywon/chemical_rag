# 引入 FastAPI 中的 APIRouter 和 HTTPException 模块，用于创建路由和处理异常
from fastapi import APIRouter, HTTPException
# 引入 Pydantic 中的 BaseModel 类，用于定义请求体的数据结构和验证
from pydantic import BaseModel
# 从数据库模块导入 execute_query 和 execute_update 函数，用于执行查询和更新操作
from database import execute_query, execute_update

# 初始化 APIRouter 实例，用于定义路由
router = APIRouter()


# 定义请求体的模型，使用 Pydantic 的 BaseModel 来验证请求的数据
class LoginRequest(BaseModel):
    # 定义 `mobile` 和 `code` 字段，分别代表手机号码和验证码
    mobile: str
    code: str


# 创建一个 POST 请求的路由，路径为 "/login/"
@router.post("/login/")
# 异步处理函数，接收 LoginRequest 类型的请求体
async def login(request: LoginRequest):
    try:
        # 执行数据库查询，验证验证码是否正确且未被使用，且是否在有效期内（15分钟内）
        result = execute_query(
            """SELECT * FROM sms_codes WHERE mobile = %s AND code = %s AND is_used = FALSE AND created_at > NOW() - INTERVAL 15 MINUTE ORDER BY created_at DESC LIMIT 1""",
            (request.mobile, request.code))

        # 如果没有找到符合条件的验证码，返回错误信息
        if not result:
            return {"code": 400, "message": "验证码错误或已过期"}

        # 获取验证码记录
        code_record = result[0]
        # 标记该验证码为已使用
        execute_update("""UPDATE sms_codes SET is_used = TRUE WHERE id = %s""", (code_record['id'],))

        # 查询用户是否已经注册
        user_result = execute_query("""SELECT * FROM users WHERE mobile = %s LIMIT 1""", (request.mobile,))

        if not user_result:
            # 如果用户不存在，则进行自动注册
            execute_update("""INSERT INTO users (mobile) VALUES (%s)""", (request.mobile,))
            # 获取新插入的用户 ID
            user_id = execute_query("SELECT LAST_INSERT_ID()")[0]['LAST_INSERT_ID()']
        else:
            # 如果用户已存在，获取用户 ID
            user_id = user_result[0]['id']
            # 更新最后登录时间
            execute_update("""UPDATE users SET last_login = NOW() WHERE id = %s""", (user_id,))

        # 返回成功登录的响应，并返回用户 ID
        return {"code": 200, "message": "登录成功", "user_id": user_id}

    # 捕获异常并返回 HTTP 500 错误，附带错误信息
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
